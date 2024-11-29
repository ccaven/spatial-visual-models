import math
import numpy as np
import torch

from fastprogress import progress_bar
from functools import partial

import wandb

from src.nanogpt.model import (
    GPTConfig,
    GPT
)

from src.vqvae.model import (
    VQVAE
)

from src.utils import compose

if __name__ == "__main__":
    device = torch.device("cuda")

    vqvae: VQVAE = torch.load("./saved_models/3_27_24/vqvae.pt")

    gpt_config = GPTConfig(
        block_size=256 + 1,
        vocab_size=1024 + 2,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=False
    )

    gpt = GPT(gpt_config)

    train_x: torch.Tensor = compose(
        lambda x: x / 255.0,
        partial(torch.permute, dims=(0, 3, 1, 2)),
        partial(torch.squeeze, dim=1),
        torch.Tensor.float,
        torch.from_numpy,
        np.load
    )("./datasets/3_25_24/frames.npy")

    vqvae = vqvae.to(device)
    gpt = gpt.to(device)

    batch_size = 4
    gradient_accumulation_steps = 10

    epochs = 20
    learning_rate = 6e-4

    optimizer = gpt.configure_optimizers(
        weight_decay=1e-1,
        learning_rate=learning_rate,
        betas=(0.9, 0.95),
        device_type=device
    )

    decay_lr = True
    warmup_iters = 200
    lr_decay_iters = 6000
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

    regenerate = False
    if regenerate:
        with torch.no_grad():
            group_size = 100
            train_i = []
            vqvae.eval()

            print("Tokenizing image dataset...\n")

            for i in progress_bar(range(0, train_x.size(0), group_size)):
                group_x = train_x[i : i + group_size].to(device)
                _, _, _, indices = vqvae.quantize(group_x)
                train_i.append(indices)
            train_i = torch.concatenate(train_i, dim=0)

            print("\nFinished tokenizing dataset.\ntrain_i.shape =", train_i.shape)
            
            np.save("./datasets/3_25_24/indices.npy", train_i.cpu().numpy())

    wandb.init(project="vqvae-v1-nanogpt")

    iter_num = 0
    for epoch in range(epochs):
        total_batch_loss = 0

        @torch.no_grad()
        def sample_imgs():
            gpt.eval()
            start_token = torch.tensor([[ 1024 ]], device=device).repeat(4, 1)
            new_tokens = gpt.generate(start_token, 256, temperature=0.9)[:, 1:]
            new_tokens = torch.minimum(new_tokens, torch.ones_like(new_tokens) * (1024 - 1))

            # un-quantize
            z_quantized = []
            for i in range(new_tokens.size(0)):
                embeddings: torch.Tensor = vqvae.vq.e_i_ts[:, new_tokens[i]]
                embeddings = embeddings.reshape(-1, 16, 16)
                z_quantized.append(embeddings)
            z_quantized = torch.stack(z_quantized)
            imgs: torch.Tensor = vqvae.decoder(z_quantized)
            imgs: np.ndarray = imgs.cpu().permute(0, 2, 3, 1).numpy()
            imgs = (imgs.clip(0.0, 1.0) * 255.0).astype(np.uint8)
            imgs = [wandb.Image(img, caption=f"Reconstructed image {i}") for i, img in enumerate(imgs)]
            gpt.train()
            return imgs

        gpt.train()
        for batch_idx, idx in enumerate(progress_bar(range(0, train_x.size(0), batch_size))):
            # Get images
            batch_x = train_x[batch_idx : batch_idx + batch_size].to(device)
            # Compute vocab
            with torch.no_grad():
                _, _, _, indices = vqvae.quantize(batch_x)
            # Pass through GPT
            start_token = torch.tensor([[ 1024 ]], device=device).repeat(batch_size, 1)
            end_token = torch.tensor([[ 1025 ]], device=device).repeat(batch_size, 1)
            tokens = torch.cat([start_token, indices, end_token], dim=1)
            x = tokens[:, :-1].contiguous()
            y = tokens[:, 1:].contiguous()
            logits, loss = gpt(x, y)
            loss.backward()
            total_batch_loss += loss.item()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr = get_lr(iter_num) if decay_lr else learning_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                iter_num += 1
                log_dict = dict()
                log_dict["batch_los"] = total_batch_loss / (batch_size * gradient_accumulation_steps)
                log_dict["learning_rate"] = lr
                if iter_num % 50 == 0:
                    log_dict["examples"] = sample_imgs()
                wandb.log(log_dict)
                total_batch_loss = 0
        
        