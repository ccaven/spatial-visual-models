# Generative models for visual and spatial information

Transforming vision and spatial tasks into language modeling problems.

Read the report: https://api.wandb.ai/links/ccaven/fzcevldh

## Directory structure

The important files are as follows:
 - `generate_block_bezier_dataset.py` constructs the training data for the VQ-VAE
 - `train_vqvae.py` actually trains the VQ-VAE
 - `generate_random_plane_2_dataset.py` constructs the training data for the transformer
 - `train_transformer_5.py` actually trains the transformer

## Why all the files?

I made quite a few attempts with various methods, so I left those methods in the repository. For example, the `src/encoder_decoder` folder and `train_encoder_decoder.py` file contains an attempt at writing a different kind of autoencoder for images where the decoder is not a convolution network but instead an autoregressive next token predictor. The `src/diffusion` contains a similar attempt where the decoder is a diffusion network.

## Credits
 - The `src/nanogpt` folder is largely taken from Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) project.
 - The `src/diffusion_2` folder is cloned from [milmor/diffusion-transformer](https://github.com/milmor/diffusion-transformer) and includes the original files and license.