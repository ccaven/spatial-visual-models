import torch
import torch.nn as nn

import math

from typing import NamedTuple

import inspect

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

class EncoderDecoderConfig(NamedTuple):
    embed_dim: int
    num_heads: int
    mlp_ratio: int
    encoder_depth: int
    decoder_depth: int
    vocab_size: int
    input_tokens: int
    dropout: float

class EncoderDecoder(nn.Module):
    def __init__(self, config: EncoderDecoderConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        self.positional_encoding = nn.Embedding(config.input_tokens, config.embed_dim)
        self.positional_encoding.weight.data = positionalencoding1d(self.config.embed_dim, config.input_tokens)
        self.positional_encoding.requires_grad_(False)

        self.transformer = nn.Transformer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            num_encoder_layers=config.encoder_depth,
            num_decoder_layers=config.decoder_depth,
            dim_feedforward=int(config.embed_dim * config.mlp_ratio),
            dropout=config.dropout,
            batch_first=True
        )

        self.decoder_head = nn.Linear(config.embed_dim, config.vocab_size)

        self.dropout = nn.Dropout(config.dropout)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.decoder_depth))

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)
        
        src = self.embedding(src) * math.sqrt(self.config.embed_dim)
        tgt = self.embedding(tgt) * math.sqrt(self.config.embed_dim)

        src = src + self.positional_encoding(torch.arange(0, src.size(1), dtype=torch.long, device=src.device))
        tgt = tgt + self.positional_encoding(torch.arange(0, tgt.size(1), dtype=torch.long, device=tgt.device))

        mem = self.transformer.encoder(src)
        
        # Apply compression to mem
        #mem = nn.functional.avg_pool1d(mem.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
        mem = mem.mean(dim=1).unsqueeze(1)

        out = self.transformer.decoder(tgt, mem, tgt_mask, tgt_is_causal=True)

        out = self.decoder_head(out)

        return out
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def regenerate(self, x: torch.Tensor, temperature: float, top_k: int | None, tokens: torch.Tensor):
        new_tokens = x.size(1) - tokens.size(1)

        src = self.embedding(x) * math.sqrt(self.config.embed_dim)
        src = src + self.positional_encoding(torch.arange(0, src.size(1), dtype=torch.long, device=src.device))
        mem = self.transformer.encoder(src)
        mem = mem.mean(dim=1).unsqueeze(1)
        
        for _ in range(new_tokens):
            tgt = self.embedding(tokens) * math.sqrt(self.config.embed_dim)
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)
        
            logits = self.transformer.decoder(tgt, mem, tgt_mask, tgt_is_causal=True)
            logits = self.decoder_head(logits)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat((tokens, idx_next), dim=1)

        return tokens

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: tuple[float, float], device: str):
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
        use_fused = fused_available and device == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer