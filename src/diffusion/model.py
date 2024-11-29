"""
Diffusion Transformer (conditional)
"""
import inspect
from functools import partial
import math
import torch
import torch.nn as nn

from typing import NamedTuple
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed

from einops import rearrange

from src.diffusion.pos_embed import get_2d_sincos_pos_embed

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    def __init__(self, embed_dim: int, freq_embed_size: int):
        super().__init__()
        self.freq_embed_size = freq_embed_size
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_size, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.freq_embed_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    

class DiTBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(embed_dim, num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=embed_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=partial(nn.GELU, approximate="tanh"),
            drop=0.0
        )
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 6 * embed_dim)
        )
    
    def forward(self, x: torch.Tensor, c: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, embed_dim, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(embed_dim, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiTConfig(NamedTuple):
    in_channels: int
    grid_size: int
    patch_size: int
    embed_dim: int
    num_heads: int
    mlp_ratio: int
    depth: int

class DiT(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        
        self.config = config
        
        assert config.grid_size % config.patch_size == 0
        
        patch_grid_size = config.grid_size // config.patch_size
        seq_len = patch_grid_size ** 2
        
        self.blocks = nn.ModuleList([
            DiTBlock(**config) for _ in range(config.depth)
        ])
        
        self.final_layer = FinalLayer(config.embed_dim, config.patch_size, config.in_channels)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, config.embed_dim), requires_grad=False)
        self.pos_embed.data.copy_(
            torch.from_numpy(
                get_2d_sincos_pos_embed(
                    config.embed_dim,
                    patch_grid_size
                )
            )
        )
        
        self.x_embedder = PatchEmbed(
            img_size=config.grid_size,
            patch_size=config.patch_size,
            in_chans=config.in_channels,
            embed_dim=config.embed_dim,
            norm_layer=nn.LayerNorm
        )
        
        self.t_embedder = TimestepEmbedder(config.embed_dim, freq_embed_size=256)
    
    def unpatchify(self, x: torch.Tensor):
        patch_grid_size = self.config.grid_size // self.config.patch_size
        x = rearrange(
            x, 
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", 
            h=patch_grid_size, 
            w=patch_grid_size, 
            p1=self.config.patch_size, 
            p2=self.config.patch_size
        )
        return x
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None):
        """
        x is (N, 4, 32, 32)
        t is (N,)
        y is (N, D) conditioning vector
        
        returns (N, 4, 32, 32)
        """
        x = self.x_embedder(x)
        t = self.t_embedder(t)
        if y is not None:
            t = t + y
        
        for block in self.blocks:
            x = block(x, t)
        
        x = self.final_layer(x, t)
        x = self.unpatchify(x)
        return x

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer