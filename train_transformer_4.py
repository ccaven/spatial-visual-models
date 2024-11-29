import math
import numpy as np
import torch

import pickle

from fastprogress import progress_bar
from functools import partial

import wandb

from src.environment.generic import SyntheticVideoDatasetConfig

from src.nanogpt.model import (
    GPTConfig,
    GPT
)

from src.nanogpt.vocab import Vocabulary

from src.vqvae.model import VQVAE

from src.utils import compose

if __name__ == "__main__":
    device = torch.device("cuda")

    vqvae: VQVAE = torch.load("./saved_models/3_26_24/vqvae.pt")

    vocab = Vocabulary()

    vocab.add_token("<sos>")
    vocab.add_token("<eos>")
    vocab.add_token_range("visual", 1024)
    vocab.add_token_range("localization-x", 8)
    vocab.add_token_range("localization-y", 8)
    localization_tokens = 2
    img_tokens = 64

    num_frames = 5

    gpt_config = GPTConfig(
        # +1 for SOS/EOS token
        block_size=(img_tokens + localization_tokens) * num_frames + 1,
        vocab_size=len(vocab),
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=False
    )

    gpt = GPT(gpt_config)

    # Preprocess dataset
    train_x: torch.Tensor = compose(
        lambda x: x / 255.0,
        partial(torch.permute, dims=(0, 1, 4, 2, 3)),
        torch.Tensor.float,
        torch.from_numpy,
        np.load
    )("./datasets/4_1_24/frames.npy")

    train_y: np.ndarray = compose(
        #torch.Tensor.float,
        #torch.from_numpy,
        np.load
    )("./datasets/4_1_24/matrices.npy")

    with open("./datasets/4_1_24/config.pkl", "rb") as handle:
        dataset_config: SyntheticVideoDatasetConfig = pickle.load(handle)

    # Produce to convert a position in space to a series of tokens
    def localization_matrix_to_token_batch(ms: np.ndarray):
        spread: float = dataset_config.trajectory_factory.args[0].spread
        ps = ms[:, :2, -1] / spread * 0.5 + 0.5
        ps = ps.clip(0.0, 1.0)
        x = ps[:, 0]
        y = ps[:, 1]
        x = (x * vocab.get_range_len("localization-x")).astype(np.int32)
        y = (y * vocab.get_range_len("localization-y")).astype(np.int32)
        x = vocab.get_token_in_range("localization-x", x)
        y = vocab.get_token_in_range("localization-y", y)
        return np.stack([x, y], axis=1)

    localization_matrix_to_token_pipeline = compose(
        partial(torch.Tensor.to, device=device),
        torch.from_numpy,
        localization_matrix_to_token_batch
    )

    vqvae = vqvae.to(device)
    gpt = gpt.to(device)

    # Training parameters
    batch_size = 4
    gradient_accumulation_steps = 10

    epochs = 50
    learning_rate = 6e-4

    decay_lr = True
    warmup_iters = 800
    lr_decay_iters = 3200
    min_lr = 6e-5

    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    optimizer = gpt.configure_optimizers(
        weight_decay=1e-1,
        learning_rate=get_lr(0),
        betas=(0.9, 0.95),
        device_type=device
    )

    # Tokenize image dataset
    regenerate_tokens = False
    if regenerate_tokens:
        with torch.no_grad():
            train_i = []
            mini_batch_size = 40
            for i in progress_bar(range(0, train_x.size(0), mini_batch_size)):
                batch_x = train_x[i : i + mini_batch_size].to(device)
                B, K, C, H, W = batch_x.shape
                batch_x = batch_x.reshape(B * K, C, H, W)
                _, _, _, indices = vqvae.quantize(batch_x)
                indices: torch.Tensor = indices.reshape(B, K, -1)
                train_i.append(indices.cpu())
            train_i = torch.concatenate(train_i, dim=0)
            print("\nFinished tokenizing dataset. train_i.shape =", train_i.shape, "\n")

        np.save("./datasets/4_1_24/indices.npy", train_i.numpy())        
    else:
        train_i = torch.from_numpy(np.load("./datasets/4_1_24/indices.npy"))

    # Training loop
    wandb.init(project="vqvae-v1-nanogpt")

    iter_num = 0
    for epoch in range(epochs):
        total_batch_loss = 0

        # Generate example images
        log_dict = dict()

        # 1. Generate all-new tokens
        with torch.no_grad():
            gpt.eval()
            start_token = torch.tensor([[ vocab.get_token("<sos>") ]], device=device).repeat(4, 1)
            new_tokens = gpt.generate(start_token, (localization_tokens + img_tokens) * num_frames + 1, temperature=0.9)[:, 1:]
            
            all_img_tokens = []
            for i in range(num_frames):
                start_idx = localization_tokens
                step_size = localization_tokens + img_tokens
                all_img_tokens.append(new_tokens[:, start_idx:][:, step_size * i:][:, :img_tokens])

            # un-quantize
            for idx, tokens in enumerate(all_img_tokens):
                tokens = tokens - vocab.get_range_start("visual")
                tokens = torch.maximum(tokens, torch.zeros_like(tokens))
                tokens = torch.minimum(tokens, torch.ones_like(tokens) * (vocab.get_range_len("visual") - 1))
                z_quantized = []
                for i in range(tokens.size(0)):
                    embeddings: torch.Tensor = vqvae.vq.e_i_ts[:, tokens[i]]
                    embeddings = embeddings.reshape(-1, 8, 8)
                    z_quantized.append(embeddings)
                z_quantized = torch.stack(z_quantized)
                imgs: torch.Tensor = vqvae.decoder(z_quantized)
                imgs: np.ndarray = imgs.cpu().permute(0, 2, 3, 1).numpy()
                imgs = (imgs.clip(0.0, 1.0) * 255.0).astype(np.uint8)
                imgs = [wandb.Image(img, caption=f"Index {i}") for i, img in enumerate(imgs)]
                log_dict[f"View {idx}"] = imgs
            
        # 2. Predict last view given prior view    
        with torch.no_grad():

            # Construct prompt
            indices = train_i[:4].to(device)
            start_token = torch.tensor([[ vocab.get_token("<sos>") ]], device=device).repeat(indices.size(0), 1)
            tokens = [start_token]
            for i in range(5):
                tokens.append(localization_matrix_to_token_pipeline(train_y[:4, i]))
                if i != 4:
                    tokens.append(vocab.get_token_in_range("visual", indices[:, i, :]))
            tokens = torch.cat(tokens, dim=1)

            # Generate new tokens
            generated = gpt.generate(tokens, 64, temperature=0.9)
            generated = generated[:, -64:]
            generated = generated - vocab.get_range_start("visual")
            generated = torch.maximum(generated, torch.zeros_like(generated))
            generated = torch.minimum(generated, torch.ones_like(generated) * (vocab.get_range_len("visual") - 1))
            
            # Decode into pixels
            z_quantized = []
            for i in range(generated.size(0)):
                embeddings: torch.Tensor = vqvae.vq.e_i_ts[:, generated[i]]
                embeddings = embeddings.reshape(-1, 8, 8)
                z_quantized.append(embeddings)
            z_quantized = torch.stack(z_quantized)
            imgs: torch.Tensor = vqvae.decoder(z_quantized)
            
            # Format to wandb.Image
            imgs: np.ndarray = imgs.cpu().permute(0, 2, 3, 1).numpy()
            imgs = (imgs.clip(0.0, 1.0) * 255.0).astype(np.uint8)
            real_imgs: np.ndarray = train_x[:4, -1].permute(0, 2, 3, 1).numpy()
            real_imgs = (real_imgs.clip(0.0, 1.0) * 255.0).astype(np.uint8)
            imgs = np.concatenate([imgs, real_imgs], axis=2)
            imgs = [wandb.Image(img, caption=f"Index {i}") for i, img in enumerate(imgs)]
            log_dict["Comparisons"] = imgs
        
        wandb.log(log_dict)

        # 3. Train epoch

        gpt.train()

        for batch_idx, idx in enumerate(progress_bar(range(0, train_x.size(0), batch_size))):
            # Retrieve training data
            batch_i = train_i[batch_idx : batch_idx + batch_size].to(device)

            # Construct sequence
            start_token = torch.tensor([[ vocab.get_token("<sos>") ]], device=device).repeat(batch_size, 1)
            end_token = torch.tensor([[ vocab.get_token("<eos>") ]], device=device).repeat(batch_size, 1)
            tokens = [start_token]
            for i in range(batch_i.size(1)):
                tokens.append(localization_matrix_to_token_pipeline(train_y[batch_idx : batch_idx + batch_size, i]))
                tokens.append(vocab.get_token_in_range("visual", batch_i[:, i, :]))
            tokens.append(end_token)
            tokens = torch.cat(tokens, dim=1)
            
            # Pass through nanoGPT model
            x = tokens[:, :-1].contiguous()
            y = tokens[:, 1:].contiguous()
            logits, loss = gpt(x, y)
            loss.backward()
            total_batch_loss += loss.item()

            # Run optimization step
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr = get_lr(iter_num) if decay_lr else learning_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                iter_num += 1
                wandb.log({
                    "batch_loss": total_batch_loss / (batch_size * gradient_accumulation_steps)
                })
                total_batch_loss = 0
    
    # Save model and configuration
    
    torch.save(gpt, "./saved_models/4_1_24/gpt.pt")
    with open("./saved_models/4_1_24/config.pkl", "wb") as handle:
        pickle.dump(gpt_config, handle)