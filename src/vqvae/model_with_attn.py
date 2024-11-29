import torch
from torch import nn

from typing import NamedTuple

from src.vqvae.model import (
    ResidualStack,
    VectorQuantizer
)

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.q = nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

def Encoder(
    in_channels: int,
    num_hiddens: int,
    num_downsampling_layers: int,
    num_residual_layers: int,
    num_residual_hiddens: int
):
    model = nn.Sequential()
    for downsampling_layer in range(num_downsampling_layers):
        if downsampling_layer == 0:
            out_channels = num_hiddens // 2
        elif downsampling_layer == 1:
            (in_channels, out_channels) = (num_hiddens // 2, num_hiddens)

        else:
            (in_channels, out_channels) = (num_hiddens, num_hiddens)

        model.add_module(
            f"down{downsampling_layer}",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )
        model.add_module(f"relu{downsampling_layer}", nn.ReLU())

    model.add_module("attn", AttnBlock(num_hiddens))

    model.add_module(
        "final_conv",
        nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            padding=1,
        ),
    )
    
    model.add_module("resstack", ResidualStack(
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens
    ))
    
    return model

def Decoder(
    embedding_dim: int,
    num_hiddens: int,
    num_upsampling_layers: int,
    num_residual_layers: int,
    num_residual_hiddens: int
):
    model = nn.Sequential()
    model.add_module("conv", nn.Conv2d(
        in_channels=embedding_dim,
        out_channels=num_hiddens,
        kernel_size=3,
        padding=1,
    ))
    model.add_module("resstack", ResidualStack(
        num_hiddens, num_residual_layers, num_residual_hiddens
    ))
    model.add_module("attn", AttnBlock(num_hiddens))
    
    for upsampling_layer in range(num_upsampling_layers):
        if upsampling_layer < num_upsampling_layers - 2:
            (in_channels, out_channels) = (num_hiddens, num_hiddens)

        elif upsampling_layer == num_upsampling_layers - 2:
            (in_channels, out_channels) = (num_hiddens, num_hiddens // 2)

        else:
            (in_channels, out_channels) = (num_hiddens // 2, 3)

        model.add_module(
            f"up{upsampling_layer}",
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )
        if upsampling_layer < num_upsampling_layers - 1:
            model.add_module(f"relu{upsampling_layer}", nn.ReLU())
    
    return model

class VQVAEWithAttnConfig(NamedTuple):
    in_channels: int
    num_hiddens: int
    num_downsampling_layers: int
    num_residual_layers: int
    num_residual_hiddens: int
    embedding_dim: int
    num_embeddings: int
    use_ema: bool
    decay: float
    epsilon: float

class VQVAEWithAttn(nn.Module):
    def __init__(
        self,
        in_channels,
        num_hiddens,
        num_downsampling_layers,
        num_residual_layers,
        num_residual_hiddens,
        embedding_dim,
        num_embeddings,
        use_ema,
        decay,
        epsilon,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels,
            num_hiddens,
            num_downsampling_layers,
            num_residual_layers,
            num_residual_hiddens,
        )
        self.pre_vq_conv = nn.Conv2d(
            in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1
        )
        self.vq = VectorQuantizer(
            embedding_dim, num_embeddings, use_ema, decay, epsilon
        )
        self.decoder = Decoder(
            embedding_dim,
            num_hiddens,
            num_downsampling_layers,
            num_residual_layers,
            num_residual_hiddens,
        )

    def quantize(self, x):
        z = self.pre_vq_conv(self.encoder(x))
        (z_quantized, dictionary_loss, commitment_loss, encoding_indices) = self.vq(z)
        return (z_quantized, dictionary_loss, commitment_loss, encoding_indices)

    def forward(self, x):
        (z_quantized, dictionary_loss, commitment_loss, _) = self.quantize(x)
        x_recon = self.decoder(z_quantized)
        return {
            "dictionary_loss": dictionary_loss,
            "commitment_loss": commitment_loss,
            "x_recon": x_recon,
        }