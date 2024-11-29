import torch
import torch.nn as nn

import inspect

from typing import NamedTuple

from src.nanogpt.model import (
    GPTConfig, 
    GPT,
    Block
)

from src.encoder_decoder.model import (
    CrossAttentionBlock
)

class EncoderConfig(NamedTuple):
    input_vocab_size: int
    output_vocab_size: int
    embed_dim: int
    num_heads: int
    mlp_ratio: int
    depth: int
    query_depth: int
    input_tokens: int
    output_tokens: int
    output_dim: int
    dropout: float

class EncoderOutput(NamedTuple):
    z_q: torch.Tensor
    dictionary_loss: torch.Tensor
    commitment_loss: torch.Tensor
    indices: torch.Tensor

class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, num_embeddings):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # Dictionary embeddings.
        limit = 3 ** 0.5
        e_i_ts = torch.FloatTensor(embedding_dim, num_embeddings).uniform_(
            -limit, limit
        )
        self.register_parameter("e_i_ts", nn.Parameter(e_i_ts))


    def forward(self, x):
        B, L, E = x.shape
        assert E == self.embedding_dim
        flat_x = x.reshape(B * L, E)
        distances = (
            (flat_x ** 2).sum(1, keepdim=True)
            - 2 * flat_x @ self.e_i_ts
            + (self.e_i_ts ** 2).sum(0, keepdim=True)
        )
        encoding_indices = distances.argmin(1).reshape(B, L)
        quantized_x = nn.functional.embedding(
            encoding_indices, self.e_i_ts.transpose(0, 1)
        ).reshape(B, L, E)

        # See second term of Equation (3).
        dictionary_loss = ((x.detach() - quantized_x) ** 2).mean()

        # See third term of Equation (3).
        commitment_loss = ((x - quantized_x.detach()) ** 2).mean()
        
        # Straight-through gradient. See Section 3.2.
        quantized_x = x + (quantized_x - x).detach()

        return (
            quantized_x,
            dictionary_loss,
            commitment_loss,
            encoding_indices,
        )

class Encoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.input_vocab_size, config.embed_dim)
        self.positional_encoding = nn.Embedding(config.input_tokens, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(GPTConfig(**{
            "n_embd": config.embed_dim,
            "n_head": config.num_heads,
            "dropout": config.dropout,
            "vocab_size": config.input_vocab_size,
            "block_size": config.input_tokens,
            "bias": False
        })) for _ in range(config.depth) ])
        
        self.query_tokens = nn.Parameter(torch.zeros(1, config.output_tokens, config.embed_dim))
        self.cross_attn = CrossAttentionBlock(config.embed_dim, config.num_heads, config.mlp_ratio, config.dropout)
        self.query_positional_encoding = nn.Embedding(config.output_tokens, config.embed_dim)
        self.query_blocks = nn.Sequential(*[Block(GPTConfig(**{
            "n_embd": config.embed_dim,
            "n_head": config.num_heads,
            "dropout": config.dropout,
            "vocab_size": config.output_vocab_size,
            "block_size": config.output_tokens,
            "bias": False
        })) for _ in range(config.query_depth) ])
        self.query_proj = nn.Linear(config.embed_dim, config.output_dim)
        self.vq = VectorQuantizer(config.output_dim, config.output_vocab_size)
    
    def forward(self, x: torch.Tensor):
        B, L = x.shape
        pos = torch.arange(0, L, dtype=torch.long, device=x.device)
        query_pos = torch.arange(0, self.config.output_tokens, dtype=torch.long, device=x.device)
        
        x = self.embedding(x)
        x = x + self.positional_encoding(pos)
        x = self.dropout(x)
        x = self.blocks(x)
        z = self.query_tokens.repeat(B, 1, 1)
        z = self.cross_attn(z, x)
        z = z + self.query_positional_encoding(query_pos)
        z = self.query_blocks(z)
        z = self.query_proj(z)
        z_q, dict_loss, commit_loss, indices = self.vq.forward(z)
        
        return EncoderOutput(z_q, dict_loss, commit_loss, indices)

class EncoderDecoderV2(nn.Module):
    def __init__(self, encoder_config: EncoderConfig, gpt_config: GPTConfig):
        super().__init__()
        self.encoder = Encoder(encoder_config)
        self.gpt = GPT(gpt_config)
    
    def forward(self, x: torch.Tensor) -> tuple[EncoderOutput, torch.Tensor, torch.Tensor]:
        """
        x is a series of tokens
        """
        enc_output: EncoderOutput = self.encoder(x)
        seq = torch.cat([enc_output.indices, x], dim=1)
        inputs = seq[:, :-1]
        targets = seq[:, 1:]
        logits, loss = self.gpt.forward(inputs, targets)
        return enc_output, logits, loss

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: tuple[float, float], device_type: str):
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