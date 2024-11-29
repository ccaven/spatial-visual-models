import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pickle

import wandb

from functools import partial

from fastprogress import progress_bar

from src.vqvae.model import (
    VQVAEConfig,
    VQVAE
)

from src.utils import compose

if __name__ == "__main__":
    device = torch.device("cuda")

    model_save_dir = "./saved_models/3_27_24"

    vqvae_config = VQVAEConfig(
        in_channels=3,
        num_hiddens=256,
        num_downsampling_layers=3,
        num_residual_layers=3,
        num_residual_hiddens=512,
        embedding_dim=128,
        num_embeddings=1024,
        use_ema=True,
        decay=0.99,
        epsilon=1e-5
    )

    model = VQVAE(*vqvae_config).to(device)

    train_x: torch.Tensor = compose(
        lambda x: x / 255.0,
        partial(torch.permute, dims=(0, 3, 1, 2)),
        partial(torch.squeeze, dim=1),
        torch.Tensor.float,
        torch.from_numpy,
        np.load
    )("./datasets/3_26_24/frames.npy")

    epochs = 10
    batch_size = 30
    gradient_accumulation_steps = 2

    commitment_loss_factor = 0.25

    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs*(1+(train_x.size(0))//(batch_size*gradient_accumulation_steps)),
    )

    wandb.init(project="vqvae-v1")
    
    test_idx = np.random.permutation(train_x.size(0))[:5]

    for epoch in range(epochs):
        batch_statistics = {
            "total_loss": 0,
            "recon_loss": 0
        }

        # Run training iterations
        model.train()
        
        for batch_idx, idx in enumerate(progress_bar(range(0, train_x.size(0), batch_size))):
            batch_x = train_x[idx : idx + batch_size].to(device)
            output = model.forward(batch_x)
            recon_loss = nn.functional.mse_loss(output["x_recon"], batch_x)

            loss = recon_loss + commitment_loss_factor * output["commitment_loss"]
            if "dictionary_loss" in output and output["dictionary_loss"] is not None:
                loss = loss + output["dictionary_loss"]
            
            loss.backward()
            
            batch_statistics["recon_loss"] += recon_loss.item()
            batch_statistics["total_loss"] += loss.item()

            if (batch_idx - 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                for key in batch_statistics.keys():
                    batch_statistics[key] /= batch_size * gradient_accumulation_steps

                wandb.log({
                    **batch_statistics,
                    "learning_rate": scheduler.get_last_lr()[-1]
                })



        # Log example images
        model.eval()
        with torch.no_grad():
            test_x = train_x[test_idx].to(device)
            test_out = model.forward(test_x)
            imgs = test_out["x_recon"]
            imgs = imgs.cpu().permute(0, 2, 3, 1).numpy()
            imgs = (imgs.clip(0.0, 1.0) * 255.0).astype(np.uint8)
            imgs = [wandb.Image(img, caption=f"Reconstructed image {i}") for i, img in enumerate(imgs)]
            wandb.log({
                "examples": imgs
            })

    wandb.finish()

    # Save model
    model = model.cpu()
    torch.save(model, model_save_dir + "/vqvae.pt")
    
    with open(model_save_dir + "/config.pkl", "wb") as handle:
        pickle.dump(vqvae_config, handle)