import copy
from functools import partial
import torch
import numpy as np
from diffusers import AutoencoderTiny
from fastprogress import progress_bar

import wandb

from collections import OrderedDict
from src.utils import compose
from src.diffusion_2.dit import DiT
from src.diffusion_2.diff_utils import (
    Diffusion
)

@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

if __name__ == "__main__":
    device = "cuda"
    
    
    # Step 1: Encode dataset with TAESD
    
    taesd: AutoencoderTiny = AutoencoderTiny.from_pretrained("madebyollin/taesd")
    taesd = taesd.to(device)
        
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
        mini_batch_size = 500
        for i in progress_bar(range(0, n_train, mini_batch_size)):
            mini_batch_x = train_x[i : i + mini_batch_size].to(device)
            train_z += [taesd.encode(mini_batch_x).latents.cpu()]
        train_z = torch.concat(train_z, dim=0)
        print(
            "\n",
            "Finished encoding dataset. train_z.shape =",
            train_z.shape,
            "\t Min max = (",
            train_z.min().item(),
            ",",
            train_z.max().item(),
            ")"
        )
    
    # Diffusion
    
    lr = 5e-3
    epochs = 20
    batch_size = 40
    gradient_accumulation_steps = 2
    
    model = DiT(
        img_size=16,
        in_channels=4,
        patch_size=1,
        depth=12,
        dim=256,
        heads=8,
        mlp_dim=1024,
        k=64
    ).to(device)
    
    ema_model = copy.deepcopy(model)
    requires_grad(ema_model, False)
    
    diffusion = Diffusion(
        P_mean=torch.mean(train_z), 
        P_std=torch.std(train_z),
        sigma_data=0.66
    )
    
    # Training
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs * n_train // (batch_size * gradient_accumulation_steps)
    )
    loss_fn = torch.nn.MSELoss()
    
    wandb.init(project="vqvae-v1-diffusion")
    for epoch in range(epochs):
        
        total_batch_loss = 0
        
        for idx, batch_idx in enumerate(progress_bar(range(0, n_train, batch_size))):
            
            # Diffusion logic            
            batch_z = train_z[idx : idx + batch_size].to(device)
            xt, t, target = diffusion.diffuse(batch_z)
            pred = model(xt, t)
            
            # Compute loss
            loss: torch.Tensor = loss_fn(pred, target)
            loss.backward()
            
            total_batch_loss += loss.item()
            
            # Update optimizer
            if (idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                #update_ema(ema_model, model, decay=0.95)
                
                log_dict = dict()
                log_dict["loss"] = total_batch_loss / (gradient_accumulation_steps * batch_size)
                log_dict["learning_rate"] = scheduler.get_last_lr()[-1]
                
                if (idx + 1) % (gradient_accumulation_steps * 20) == 0:
                    with torch.no_grad():
                        gen_batch_size = 4
                        gen_batch = diffusion.sample(model, (gen_batch_size, 4, 16, 16))
                        print("\nGenerated batch minmax:", gen_batch.min(), gen_batch.max())
                        decoded = taesd.decode(gen_batch).sample
                        decoded = decoded.cpu().clip(0.0, 1.0).permute(0, 2, 3, 1)
                        decoded: np.ndarray = decoded.numpy()
                        decoded = (decoded * 255.0).astype(np.uint8)
                        imgs = [wandb.Image(img, caption=f"Reconstructed image {i}") for i, img in enumerate(decoded)]
                        log_dict["images"] = imgs
                
                wandb.log(log_dict)
                
                total_batch_loss = 0

    wandb.finish()