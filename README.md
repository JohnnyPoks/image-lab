# image-lab

A small study in generative deep learning: GANs trained from scratch and compared across three adversarial objectives, plus image restoration, wrapped in one deployed Gradio app.

## What it does

**Synthesis** — for each dataset a GAN is trained from scratch on multiple objectives (**DCGAN, LSGAN, WGAN-GP**), compared on FID / Inception Score, and the winner is served:

- **Generate Digit** — MNIST 28×28 (DCGAN vs LSGAN vs WGAN-GP)
- **Generate Anime Face** — anime faces 64×64 (DCGAN vs LSGAN)

**Restoration** —

- **Upscale** — 4× super-resolution with Real-ESRGAN (general or anime model)

The app checks which trained weights are present at start-up and only exposes those features.

## Live demo

[huggingface.co/spaces/PokamJohn/image-lab](https://huggingface.co/spaces/PokamJohn/image-lab)

## Key results

| Dataset | Winner | Best FID | Note |
|---|---|---|---|
| MNIST (30 ep) | **WGAN-GP** | 15.6 | best from epoch 1; DCGAN 22.3, LSGAN 24.5 |
| Anime faces (60 ep) | **DCGAN** | 69.2 | DCGAN converges much faster than LSGAN (166.2) |

## Tech stack

- PyTorch / torchvision
- Custom DCGAN / LSGAN / WGAN-GP (generator, discriminator, critic — trained from scratch)
- Real-ESRGAN (basicsr) for super-resolution
- Gradio + Hugging Face Spaces

## Running locally

```bash
git clone https://github.com/JohnnyPoks/image-lab.git
cd image-lab
pip install -r requirements.txt
python app.py
```

Drop trained weights into `weights/` before launching — see the notebooks for how each one is produced.

## Notebooks

- [01_realesrgan_inference.ipynb](notebooks/01_realesrgan_inference.ipynb) — Real-ESRGAN super-resolution; PSNR/SSIM benchmark
- [02_dcgan_mnist.ipynb](notebooks/02_dcgan_mnist.ipynb) — DCGAN vs LSGAN vs WGAN-GP on MNIST (full three-way comparison)
- [04_dcgan_anime.ipynb](notebooks/04_dcgan_anime.ipynb) — DCGAN vs LSGAN on 64×64 anime faces — the main synthesis result
- [03_dcgan_cifar10.ipynb](notebooks/03_dcgan_cifar10.ipynb) — CIFAR-10 experiment (reproducible extension)
- [05_dcgan_celeba.ipynb](notebooks/05_dcgan_celeba.ipynb) — CelebA face generation (reproducible extension)

Every notebook is self-contained and runs on Colab or Kaggle. Each trains the GAN objectives on its dataset, compares them on FID / Inception Score, and promotes the winner. Shared engine: mixed-precision training (version-safe AMP), a WGAN-GP critic with gradient penalty, per-architecture checkpoint-resume, D(x)/D(G(z)) tracking, and training-history logging to JSON.
