
from typing import NamedTuple

import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=12,
        qkv_bias=False,
        use_sdpa=True
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, int(dim*2), bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_sdpa = use_sdpa

    def forward(self, q, x):
        B, n, C = q.shape
        q = self.q(q).reshape(B, n, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        B, N, C = x.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (batch_size, num_heads, seq_len, feature_dim_per_head)

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                q = nn.functional.scaled_dot_product_attention(q, k, v)
        else:
            xattn = (q @ k.transpose(-2, -1)) * self.scale
            xattn = xattn.softmax(dim=-1)  # (batch_size, num_heads, query_len, seq_len)
            q = (xattn @ v)

        q = q.transpose(1, 2).reshape(B, n, C)
        q = self.proj(q)
    
        return q

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.xattn = CrossAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, q, x):
        y = self.xattn(q, self.norm1(x))
        q = q + y
        q = q + self.mlp(self.norm2(q))
        return q

class EncoderDecoderConfig(NamedTuple):
    embed_dim: int
    num_heads: int
    mlp_ratio: float
    encoder_depth: int
    query_depth: int
    decoder_depth: int
    input_vocab_size: int
    input_tokens: int
    # output_vocab_size: int
    output_tokens: int
    dropout: float

class EncoderDecoder(nn.Module):
    def __init__(self, config: EncoderDecoderConfig):
        super().__init__()

        self.config = config

        self.encoder_embedding = nn.Embedding(config.input_vocab_size, config.embed_dim)

        self.encoder_positional_encoding = nn.Embedding(config.input_tokens, config.embed_dim)

        self.encoder_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.embed_dim,
                nhead=config.num_heads,
                dim_feedforward=int(config.embed_dim * config.mlp_ratio),
                dropout=config.dropout,
                layer_norm_eps=1e-6,
                batch_first=True
            ),
            num_layers=config.encoder_depth
        )

        self.query_tokens = nn.Parameter(torch.zeros(1, config.output_tokens, config.embed_dim))

        self.cross_attention = CrossAttentionBlock(config.embed_dim, config.num_heads, mlp_ratio=config.mlp_ratio, dropout=config.dropout)

        self.query_pos_encoding = nn.Embedding(config.output_tokens, config.embed_dim)

        self.query_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.embed_dim,
                nhead=config.num_heads,
                dim_feedforward=int(config.embed_dim * config.mlp_ratio),
                dropout=config.dropout,
                batch_first=True
            ),
            num_layers=config.query_depth
        )

        self.decoder_embedding = nn.Embedding(config.input_vocab_size + 2, config.embed_dim)

        self.decoder_positional_encoding = nn.Embedding(config.input_tokens + 1, config.embed_dim)

        self.decoder_blocks = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.embed_dim,
                nhead=config.num_heads,
                dim_feedforward=int(config.embed_dim * config.mlp_ratio),
                dropout=config.dropout,
                layer_norm_eps=1e-6,
                batch_first=True
            ),
            num_layers=config.decoder_depth
        )

        self.decoder_head = nn.Linear(config.embed_dim, config.input_vocab_size + 2)

        self.decoder_embedding.weight = self.decoder_head.weight

        self.dropout = nn.Dropout(config.dropout)
    
    def encode(self, x: torch.Tensor):
        """
        x is a Tensor of shape (B, L)
        """

        B, L = x.shape

        pos = torch.arange(0, L, dtype=torch.long, device=x.device)

        #enc_emb = self.encoder_embedding(x)
        enc_emb = self.decoder_embedding(x)
        enc_emb = enc_emb + self.encoder_positional_encoding(pos)
        enc_emb = self.dropout(enc_emb)
        enc_emb = self.encoder_blocks(enc_emb)

        q_pos = torch.arange(0, self.config.output_tokens, dtype=torch.long, device=x.device)

        z = self.query_tokens.repeat(B, 1, 1)
        z = self.cross_attention(z, enc_emb)
        z = z + self.query_pos_encoding(q_pos)
        z = self.query_blocks(z)

        return z
    
    def decode(self, x: torch.Tensor, z: torch.Tensor | None):
        B, L = x.shape

        pos = torch.arange(0, L + 1, dtype=torch.long, device=x.device)
        
        sos_token = torch.tensor([[ self.config.input_vocab_size ]], dtype=torch.long, device=x.device).repeat(B, 1)
        eos_token = torch.tensor([[ self.config.input_vocab_size + 1 ]], dtype=torch.long, device=x.device).repeat(B, 1)

        seq = torch.cat([
            sos_token,
            x,
            eos_token
        ], dim=1)

        inputs = seq[:, :-1]
        target = seq[:, 1:]

        x = self.decoder_embedding(inputs)
        x = x + self.decoder_positional_encoding(pos)
        x = self.dropout(x)

        if z is not None:
            x = self.decoder_blocks(
                x, 
                z, 
                tgt_mask=nn.Transformer.generate_square_subsequent_mask(L + 1, device=x.device),
                tgt_is_causal=True
            )
        
        if z is None:
            z = torch.zeros((B, 1, self.config.embed_dim), device=x.device, dtype=x.dtype)
            #z_mask = torch.ones((B, 1), device=x.device).bool()
            
            x = self.decoder_blocks(
                x, 
                z,
                tgt_mask=nn.Transformer.generate_square_subsequent_mask(L + 1, device=x.device),
                tgt_is_causal=True,
                #memory_key_padding_mask=z_mask
            )
        
        logits = self.decoder_head(x)

        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target.reshape(-1), ignore_index=-1)

        return logits, loss

    def forward_unsupervised(self, x: torch.Tensor):
        logits, loss = self.decode(x, z=None)
        return logits, loss

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        logits, loss = self.decode(x, z)
        return z, logits, loss
    
    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, temperature: float, top_k: int | None):
        B = x.size(0)
        
        q = self.encode(x)

        tokens = torch.tensor([[ self.config.input_vocab_size ]], dtype=torch.long, device=x.device).repeat(B, 1)

        for _ in range(self.config.input_tokens):
            # Generate new token
            logits, _ = self.decode(tokens, q)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat((tokens, idx_next), dim=1)
        
        return tokens[:, 1:]
    
    def generate_unsupervised(self, batch_size: int, device: str, temperature: float, top_k: int | None):
        B = batch_size

        tokens = torch.tensor([[ self.config.input_vocab_size ]], dtype=torch.long, device=device).repeat(B, 1)

        for _ in range(self.config.input_tokens):
            # Generate new token
            logits, _ = self.forward_unsupervised(tokens)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat((tokens, idx_next), dim=1)
        
        return tokens[:, 1:]