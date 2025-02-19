# vqvae_compression

Code for [my blog post](https://ovgu-ailab.github.io/blog/methods/2024/05/28/vqvae-compression.html) investigating compression behavior in (residual) VQ-VAEs. Made public on request.

## Rough structure

- aldi: Main project files/modules
  - modeling: Stuff to build models!
    - layers: `tf.keras.Layer` instances that can be broadly useful
    - callbacks: Guess what!
    - vq: VQ-VAE model code
- experiment_novq: Notebook running basic autoencoders with different latent dimension `d`
- experiment_vq_d4: Notebook running VQ-VAE with `d=4`, but different codecook sizes.

Residual VQ can be achieved by simpy increasing `n_levels` in the `RVQ` object.

It's called `aldi` because I took this from a larger project on Audio Latent DIfusion. I had trouble with the RVQ-VAEs there and wanted to investigate simpler examples. Code is messy and overengineered. Sorry!
