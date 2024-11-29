from functools import partial

import numpy as np

import torch

from diffusers import AutoencoderTiny

from fastprogress import progress_bar

from src.diffusion.model import (
    DiTConfig,
    DiT
)

from src.utils import compose

if __name__ == "__main__":
    
    device = torch.device("cuda")
    
    taesd: AutoencoderTiny = AutoencoderTiny.from_pretrained("madebyollin/taesd")
    taesd = taesd.to(device)
    
    """ Step 1: Encode dataset with TAESD """
    
    train_x: torch.Tensor = compose(
        lambda x: x / 255.0,
        partial(torch.permute, dims=(0, 3, 1, 2)),
        partial(torch.squeeze, dim=1),
        torch.Tensor.float,
        torch.from_numpy,
        np.load
    )("./datasets/3_25_24/frames.npy")
    
    n_train = train_x.size(0)
    
    with torch.no_grad():
        train_z = []
        for i in progress_bar(range(0, n_train, 5000)):
            mini_batch_x = train_x[i : i + 200].to(device)
            train_z += [taesd.encode(mini_batch_x).latents]
        train_z = torch.concat(train_z, dim=0)
        print("Finished encoding dataset. train_z.shape =", train_z.shape)
    
    """ Step 2: Create DiT model """
    
    config = DiTConfig(
        in_channels=4,
        grid_size=16,
        patch_size=4,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=3,
        depth=6
    )
    
    model = DiT(config)
    model = model.to(device)
    
    """ Step 3: Configure optimizers """
    
    num_timesteps = 1000
    
    epochs = 10
    batch_size = 4
    gradient_accumulation_steps = 10
    
    optimizer = None    
    
    """ Step 4: Train loop """
    
    for epoch in range(epochs):
        for batch_idx, idx in enumerate(progress_bar(range(0, n_train, batch_size))):
            batch_x = train_z[idx : idx + batch_size]
            
            # Timesteps
            t = torch.randint(0, num_timesteps, (batch_size,), device=device)
            
            # Compute noise
            noise = None
            noisy_x = None
            
            # Run prediction
            pred_noise = model.forward(noisy_x, t, y = None)
            
            loss = torch.nn.functional.mse_loss(
                pred_noise,
                noise
            )
            
            loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()