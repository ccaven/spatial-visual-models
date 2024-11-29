import math
import numpy as np
import torch

import wandb

from functools import partial

from fastprogress import progress_bar

from src.vqvae.model import (
    VQVAE
)

from src.encoder_decoder_3.model import (
    EncoderDecoderConfig,
    EncoderDecoder
)

from src.utils import compose

def add_sos_eos(indices: torch.Tensor, sos_token: int, eos_token: int | None):
    B, _ = indices.shape
    sos = torch.tensor([[ sos_token ]], device=indices.device, dtype=torch.long).repeat(B, 1)
    if eos_token is None:
        return torch.cat([
            sos,
            indices
        ], dim=1)
    
    eos = torch.tensor([[ eos_token ]], device=indices.device, dtype=torch.long).repeat(B, 1)
    return torch.cat([
        sos,
        indices,
        eos
    ], dim=1)

if __name__ == "__main__":
    device = torch.device("cuda")

    vqvae: VQVAE = torch.load("./saved_models/3_27_24/vqvae.pt")

    img_vocab_size = 1024

    encoder_decoder_config = EncoderDecoderConfig(
        embed_dim=512,
        num_heads=8,
        mlp_ratio=3,
        encoder_depth=6,
        decoder_depth=12,
        vocab_size=img_vocab_size + 2,
        input_tokens=256 + 2,
        dropout=0.1
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

    batch_size = 4
    gradient_accumulation_steps = 10

    epochs = 5
    learning_rate = 6e-4

    optimizer = encoder_decoder.configure_optimizers(
        weight_decay=1e-1,
        learning_rate=learning_rate,
        betas=(0.9, 0.95),
        device=device
    )

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=6e-4,
        epochs=epochs,
        steps_per_epoch=(train_x.size(0)//(gradient_accumulation_steps * batch_size)),
        pct_start=0.1
    )

    # 1. Turn dataset into tokens
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
    else:
        train_i = compose(
            partial(torch.Tensor.to, device=device),
            torch.from_numpy,
            np.load
        )("./datasets/3_25_24/indices.npy")
        
        print(train_i.min(), train_i.max())
    
    @torch.no_grad()
    def sample_images():
        encoder_decoder.eval()

        test_i = train_i[:5]

        encoder_input = add_sos_eos(
            test_i, 
            sos_token=img_vocab_size, 
            eos_token=img_vocab_size + 1
        )

        reconstructed = encoder_decoder.regenerate(
            encoder_input,
            temperature=1.0,
            top_k=None,
            tokens=torch.tensor([[ img_vocab_size ]], dtype=torch.long, device=device).repeat(encoder_input.size(0), 1)
        )

        new_tokens = reconstructed[:, 1:-1]
        new_tokens = torch.minimum(new_tokens, torch.ones_like(new_tokens) * (img_vocab_size - 1))

        z_q = []
        for i in range(test_i.size(0)):
            emb: torch.Tensor = vqvae.vq.e_i_ts[:, new_tokens[i]]
            emb = emb.reshape(-1, 16, 16)
            z_q.append(emb)
        z_q = torch.stack(z_q, dim=0)
        imgs: torch.Tensor = vqvae.decoder(z_q)
        imgs: np.ndarray = imgs.cpu().permute(0, 2, 3, 1).numpy()
        
        real_imgs: np.ndarray = train_x[:5].cpu().permute(0, 2, 3, 1).numpy()
        imgs = np.concatenate([imgs, real_imgs], axis=2)        
        imgs = (imgs.clip(0.0, 1.0) * 255.0).astype(np.uint8)
        imgs = [wandb.Image(img, caption=f"Sample {i}") for i, img in enumerate(imgs)]

        encoder_decoder.train()

        return imgs

    wandb.init(project="vqvae-v1-encoder-decoder")

    iter_num = 0
    for epoch in range(epochs):
        total_batch_loss = 0
        for idx, i in enumerate(progress_bar(range(0, train_i.size(0), batch_size))):
            batch_i = train_i[i : i + batch_size]
            encoder_input = add_sos_eos(
                batch_i, 
                sos_token=img_vocab_size, 
                eos_token=img_vocab_size + 1
            )

            src = encoder_input[:, :-1]
            tgt = encoder_input[:, 1:]

            logits: torch.Tensor = encoder_decoder(src, tgt)
            
            loss_weight = torch.linspace(2.0, -2.0, steps=logits.size(1), device=logits.device).exp().unsqueeze(0).repeat(logits.size(0), 1).flatten()

            loss = torch.nn.functional.cross_entropy(
                input=logits.reshape(-1, logits.size(-1)),
                target=tgt.reshape(-1),
                #ignore_index=-1,
                reduction="none"                
            )
            loss = loss * loss_weight
            #loss = loss.reshape(src.size(0), -1)[0, :].mean()
            loss = loss.sum()
            loss.backward()

            total_batch_loss += loss.item()
            if (idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                log_dict = {
                    "batch_loss": total_batch_loss / (batch_size * gradient_accumulation_steps),
                    "learning_rate": lr_scheduler.get_last_lr()[-1]
                }

                if (iter_num + 1) % 25 == 0:
                    log_dict["examples"] = sample_images()
                    
                iter_num += 1

                wandb.log(log_dict)
                total_batch_loss = 0
    
    torch.save(encoder_decoder, "./saved_models/3_29_24/encoder_decoder_3.pt")

    wandb.finish()