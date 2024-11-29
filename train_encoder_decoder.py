import math
import numpy as np
import torch

import wandb

from functools import partial

from fastprogress import progress_bar

from src.vqvae.model import (
    VQVAE
)

from src.encoder_decoder.model import (
    EncoderDecoderConfig,
    EncoderDecoder
)

from src.utils import compose

if __name__ == "__main__":
    device = torch.device("cuda")

    vqvae: VQVAE = torch.load("./saved_models/3_27_24/vqvae.pt")

    encoder_decoder_config = EncoderDecoderConfig(
        embed_dim=768,
        num_heads=12,
        mlp_ratio=3,
        encoder_depth=12,
        query_depth=6,
        decoder_depth=12,
        input_vocab_size=1024,
        input_tokens=256,
        output_tokens=64,
        dropout=0.0
    )

    encoder_decoder = EncoderDecoder(encoder_decoder_config)

    train_x: torch.Tensor = compose(
        lambda x: x / 255.0,
        partial(torch.permute, dims=(0, 3, 1, 2)),
        partial(torch.squeeze, dim=1),
        torch.Tensor.float,
        torch.from_numpy,
        np.load
    )("./datasets/3_25_24/frames.npy")

    vqvae = vqvae.to(device)
    encoder_decoder = encoder_decoder.to(device)

    supervised = False

    batch_size = 4
    gradient_accumulation_steps = 10

    epochs = 5
    learning_rate = 6e-4

    optimizer = torch.optim.Adam(
        encoder_decoder.parameters(),
        lr=6e-5,
        betas=(0.9, 0.95)
    )

    decay_lr = True
    warmup_iters = 100
    lr_decay_iters = 1000
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
    
    wandb.init(project="vqvae-v1-encoder-decoder")

    iter_num = 0

    # 1. Turn dataset into tokens
    regenerate = True
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

    @torch.no_grad()
    def sample_images_supervised():
        encoder_decoder.eval()

        test_i = train_i[:5]
        recon_i = encoder_decoder.reconstruct(test_i, temperature=0.9, top_k=None)
        recon_i = torch.minimum(recon_i, torch.ones_like(recon_i) * (encoder_decoder.config.input_vocab_size - 1))
        z_q = []
        for i in range(test_i.size(0)):
            emb: torch.Tensor = vqvae.vq.e_i_ts[:, recon_i[i]]
            emb = emb.reshape(-1, 16, 16)
            z_q.append(emb)
        z_q = torch.stack(z_q, dim=0)
        imgs: torch.Tensor = vqvae.decoder(z_q)
        imgs: np.ndarray = imgs.cpu().permute(0, 2, 3, 1).numpy()
        
        real_imgs: np.ndarray = train_x[:5].cpu().permute(0, 2, 3, 1).numpy()
        real_imgs = (real_imgs.clip(0.0, 1.0) * 255.0).astype(np.uint8)
        imgs = (imgs.clip(0.0, 1.0) * 255.0).astype(np.uint8)
        imgs = np.concatenate([imgs, real_imgs], axis=2)
        imgs = [wandb.Image(img, caption=f"Reconstructed image {i}") for i, img in enumerate(imgs)]

        encoder_decoder.train()

        return imgs

    @torch.no_grad()
    def sample_images_unsupervised():
        encoder_decoder.eval()
        num_samples = 5
        recon_i = encoder_decoder.generate_unsupervised(batch_size=num_samples, device=train_i.device,temperature=0.9, top_k=None)
        recon_i = torch.minimum(recon_i, torch.ones_like(recon_i) * (encoder_decoder.config.input_vocab_size - 1))
        z_q = []
        for i in range(num_samples):
            emb: torch.Tensor = vqvae.vq.e_i_ts[:, recon_i[i]]
            emb = emb.reshape(-1, 16, 16)
            z_q.append(emb)
        z_q = torch.stack(z_q, dim=0)
        imgs: torch.Tensor = vqvae.decoder(z_q)
        imgs: np.ndarray = imgs.cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs.clip(0.0, 1.0) * 255.0).astype(np.uint8)
        imgs = [wandb.Image(img, caption=f"Generated image {i}") for i, img in enumerate(imgs)]

        encoder_decoder.train()

        return imgs

    for epoch in range(epochs):
        total_batch_loss = 0

        encoder_decoder.train()
        for idx, i in enumerate(progress_bar(range(0, train_i.size(0), batch_size))):
            batch_i = train_i[i : i + batch_size]
            if supervised:
                q, logits, loss = encoder_decoder.forward(batch_i)
                loss.backward()
            else:
                logits, loss = encoder_decoder.forward_unsupervised(batch_i)
                loss.backward()
            total_batch_loss += loss.item()
            if (idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr = get_lr(iter_num) if decay_lr else learning_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                iter_num += 1
                log_dict = {
                    "batch_loss": total_batch_loss / (batch_size * gradient_accumulation_steps),
                    "learning_rate": lr
                }

                if iter_num % 10 == 0:
                    log_dict["examples"] = (sample_images_supervised if supervised else sample_images_unsupervised)()

                wandb.log(log_dict)
                total_batch_loss = 0
    
    torch.save(encoder_decoder, "./saved_models/3_27_24_2/encoder_decoder.pt")