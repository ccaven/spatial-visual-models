import numpy as np
import torch

from fastprogress import progress_bar
from functools import partial

import wandb

from src.vqvae.model import (
    VQVAE
)

from src.encoder_decoder_2.model import (
    EncoderConfig,
    GPTConfig,
    EncoderDecoderV2,
    EncoderOutput
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
    
    img_tokens = 256
    img_vocab_size = 231
    
    encoder_config = EncoderConfig(
        input_vocab_size=img_vocab_size + 2,
        output_vocab_size=512,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=3,
        depth=6,
        query_depth=6,
        input_tokens=img_tokens + 2,
        output_tokens=32,
        output_dim=256,
        dropout=0.0
    )
    
    gpt_config = GPTConfig(
        block_size=img_tokens + encoder_config.output_tokens + 1,
        vocab_size=img_vocab_size + encoder_config.output_vocab_size + 2,
        n_layer=12,
        n_embd=768,
        dropout=0.0,
        bias=False
    )

    vqvae = vqvae.to(device)
    encoder_decoder = EncoderDecoderV2(encoder_config, gpt_config).to(device)
    
    train_x: torch.Tensor = compose(
        lambda x: x / 255.0,
        partial(torch.permute, dims=(0, 3, 1, 2)),
        partial(torch.squeeze, dim=1),
        torch.Tensor.float,
        torch.from_numpy,
        np.load
    )("./datasets/3_25_24/frames.npy")
    
    batch_size = 4
    gradient_accumulation_steps = 10

    epochs = 5
    learning_rate = 6e-4
    
    optimizer = encoder_decoder.configure_optimizers(
        weight_decay=1e-1,
        learning_rate=6e-5,
        betas=(0.9, 0.95),
        device_type=device
    )
    
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=6e-4,
        epochs=epochs,
        steps_per_epoch=(train_x.size(0)//(gradient_accumulation_steps * batch_size)),
        pct_start=0.1
    )
    
    wandb.init(project="vqvae-v1-encoder-decoder")

    iter_num = 0

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
        
        encoder_output: EncoderOutput = encoder_decoder.encoder(encoder_input)
        
        gpt_input = add_sos_eos(
            encoder_output.indices,
            sos_token=img_vocab_size + encoder_config.output_tokens,
            eos_token=None
        )
        
        new_tokens = encoder_decoder.gpt.generate(gpt_input, max_new_tokens=img_tokens, temperature=0.9)
        new_tokens = new_tokens[:, -img_tokens:]
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
        imgs = [wandb.Image(img, caption=f"Reconstructed image {i}") for i, img in enumerate(imgs)]

        encoder_decoder.train()
        
        return imgs
    
    wandb.log({
        "examples": sample_images()
    })
    
    for epoch in range(epochs):
        total_batch_loss = 0
        for idx, i in enumerate(progress_bar(range(0, train_i.size(0), batch_size))):
            batch_i = train_i[i : i + batch_size]
            
            encoder_output: EncoderOutput = encoder_decoder.encoder(
                add_sos_eos(
                    batch_i, 
                    sos_token=img_vocab_size, 
                    eos_token=img_vocab_size + 1
                )
            )
            
            gpt_seq = add_sos_eos(
                encoder_output.indices,
                sos_token=img_vocab_size + encoder_config.output_tokens,
                eos_token=img_vocab_size + encoder_config.output_tokens + 1
            )
            
            inputs = gpt_seq[:, :-1].contiguous()
            target = gpt_seq[:, 1:].contiguous()
            
            logits, loss = encoder_decoder.gpt(
                idx=inputs,
                targets=target
            )
            
            loss = loss + 0.1 * encoder_output.commitment_loss + 0.1 * encoder_output.dictionary_loss
            
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

                if (iter_num + 1) % 50 == 0:
                    log_dict["examples"] = sample_images()
                    
                iter_num += 1

                wandb.log(log_dict)
                total_batch_loss = 0
    
    torch.save(encoder_decoder, "./saved_models/3_28_24_2/encoder_decoder_v2.pt")