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

    vqvae: VQVAE = torch.load("./saved_models/3_26_24/vqvae.pt")

    gpt_config = GPTConfig(
        block_size=65 * 3 + 1,
        vocab_size=1024 + 2 + 3,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=False
    )

    gpt = GPT(gpt_config)

    train_x: torch.Tensor = compose(
        lambda x: x / 255.0,
        partial(torch.permute, dims=(0, 1, 4, 2, 3)),
        torch.Tensor.float,
        torch.from_numpy,
        np.load
    )("./datasets/3_26_24_3/frames.npy")

    vqvae = vqvae.to(device)
    gpt = gpt.to(device)

    batch_size = 2
    gradient_accumulation_steps = 20

    epochs = 30
    learning_rate = 6e-4

    optimizer = gpt.configure_optimizers(
        weight_decay=1e-1,
        learning_rate=learning_rate,
        betas=(0.9, 0.95),
        device_type=device
    )

    decay_lr = True
    warmup_iters = 2000
    lr_decay_iters = 600000
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

    wandb.init(project="vqvae-v1-nanogpt")

    iter_num = 0
    for epoch in range(epochs):
        total_batch_loss = 0

        # Generate example images
        log_dict = dict()

        with torch.no_grad():
            # 1. Generate all-new tokens
            gpt.eval()
            start_token = torch.tensor([[ 1024 ]], device=device).repeat(4, 1)
            new_tokens = gpt.generate(start_token, 65 * 3 + 1, temperature=0.9)[:, 1:]
            
            img1_tokens = new_tokens[:, 2          :2+64]
            img2_tokens = new_tokens[:, 2+64+1     :2+64+1+64]
            img3_tokens = new_tokens[:, 2+64+1+64+1:2+64+1+64+1+64]

            # un-quantize
            
            for idx, img_tokens in enumerate([img1_tokens, img2_tokens, img3_tokens]):
                img_tokens = torch.minimum(img_tokens, torch.ones_like(img_tokens) * 1023)
                z_quantized = []
                for i in range(img_tokens.size(0)):
                    embeddings: torch.Tensor = vqvae.vq.e_i_ts[:, img_tokens[i]]
                    embeddings = embeddings.reshape(-1, 8, 8)
                    z_quantized.append(embeddings)
                z_quantized = torch.stack(z_quantized)
                imgs: torch.Tensor = vqvae.decoder(z_quantized)
                imgs: np.ndarray = imgs.cpu().permute(0, 2, 3, 1).numpy()
                imgs = (imgs.clip(0.0, 1.0) * 255.0).astype(np.uint8)
                imgs = [wandb.Image(img, caption=f"Index {i}") for i, img in enumerate(imgs)]
                log_dict[f"View {idx}"] = imgs
            
            # 2. Predict last view given prior views
            batch_x = train_x[:4].to(device)
            B, K = batch_x.shape[:2]
            batch_x = batch_x.reshape(B * K, 3, 128, 128)
            _, _, _, indices = vqvae.quantize(batch_x)
            batch_x = batch_x.reshape(B, K, 3, 128, 128)
            indices: torch.Tensor = indices.reshape(B, K, 64)
            
            start_token = torch.tensor([[ 1024 ]], device=device).repeat(B, 1)
            
            view1_token = torch.tensor([[ 1026 ]], device=device).repeat(B, 1)
            view2_token = torch.tensor([[ 1027 ]], device=device).repeat(B, 1)
            view3_token = torch.tensor([[ 1028 ]], device=device).repeat(B, 1)
            
            img1_tokens = indices[:, 0, :]
            img2_tokens = indices[:, 1, :]
            img3_tokens = indices[:, 2, :]
            
            tokens = torch.cat([
                start_token,
                view1_token,
                img1_tokens,
                view3_token,
                img3_tokens,
                view2_token,
            ], dim=1)

            generated = gpt.generate(tokens, 64, temperature=0.9)

            img_tokens = generated[:, -64:]
            img_tokens = torch.minimum(img_tokens, torch.ones_like(img_tokens) * 1023)
            z_quantized = []
            for i in range(img_tokens.size(0)):
                embeddings: torch.Tensor = vqvae.vq.e_i_ts[:, img_tokens[i]]
                embeddings = embeddings.reshape(-1, 8, 8)
                z_quantized.append(embeddings)
            z_quantized = torch.stack(z_quantized)
            imgs: torch.Tensor = vqvae.decoder(z_quantized)
            imgs: np.ndarray = imgs.cpu().permute(0, 2, 3, 1).numpy()
            imgs = (imgs.clip(0.0, 1.0) * 255.0).astype(np.uint8)
            real_imgs: np.ndarray = batch_x.cpu()[:, 1].permute(0, 2, 3, 1).numpy()
            real_imgs = (real_imgs.clip(0.0, 1.0) * 255.0).astype(np.uint8)
            imgs = np.concatenate([imgs, real_imgs], axis=2)
            imgs = [wandb.Image(img, caption=f"Index {i}") for i, img in enumerate(imgs)]
            log_dict[f"Comparisons"] = imgs
        
        wandb.log(log_dict)

        gpt.train()
        for batch_idx, idx in enumerate(progress_bar(range(0, train_x.size(0), batch_size))):
            # Get images
            batch_x = train_x[batch_idx : batch_idx + batch_size].to(device)
            # Compute vocab
            with torch.no_grad():
                B, K = batch_x.shape[:2]
                batch_x = batch_x.reshape(B * K, 3, 128, 128)
                _, _, _, indices = vqvae.quantize(batch_x)
                indices: torch.Tensor = indices.reshape(B, K, 64)
            # Pass through GPT
            start_token = torch.tensor([[ 1024 ]], device=device).repeat(batch_size, 1)
            
            view1_token = torch.tensor([[ 1026 ]], device=device).repeat(batch_size, 1)
            view2_token = torch.tensor([[ 1027 ]], device=device).repeat(batch_size, 1)
            view3_token = torch.tensor([[ 1028 ]], device=device).repeat(batch_size, 1)
            
            img1_tokens = indices[:, 0, :]
            img2_tokens = indices[:, 1, :]
            img3_tokens = indices[:, 2, :]

            end_token = torch.tensor([[ 1025 ]], device=device).repeat(batch_size, 1)
            
            tokens = torch.cat([
                start_token,
                view1_token,
                img1_tokens,
                view3_token,
                img3_tokens,
                view2_token,
                img2_tokens,
                end_token
            ], dim=1)
            
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
                wandb.log({
                    "batch_loss": total_batch_loss / (batch_size * gradient_accumulation_steps)
                })
                total_batch_loss = 0
        
        