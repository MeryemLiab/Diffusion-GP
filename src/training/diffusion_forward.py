# %%
import torch


def linear_beta_schedule(num_steps, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, num_steps)


def prepare_noise_schedule(num_steps, beta_start=1e-4, beta_end=2e-2):
    betas = linear_beta_schedule(num_steps, beta_start, beta_end)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def q_sample_images(x0, t, alpha_bars):
    noise = torch.randn_like(x0)
    alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
    x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise
    return x_t, noise


# %%
import sys
import os
sys.path.append(os.path.abspath(".."))

import torch
import matplotlib.pyplot as plt

from src.datasets.mnist import get_mnist_dataloader

dataloader = get_mnist_dataloader(batch_size=8)
images, labels = next(iter(dataloader))

num_steps = 100
_, _, alpha_bars = prepare_noise_schedule(num_steps)

t = torch.randint(0, num_steps, (images.shape[0],))

x_t, noise = q_sample_images(images, t, alpha_bars)

print(images.shape)
print(x_t.shape)
print(noise.shape)


# %%
import torch
import matplotlib.pyplot as plt

from src.datasets.mnist import get_mnist_dataloader
from src.training.diffusion_forward import prepare_noise_schedule, q_sample_images

dataloader = get_mnist_dataloader(batch_size=8)
images, labels = next(iter(dataloader))

num_steps = 100
_, _, alpha_bars = prepare_noise_schedule(num_steps)

t = torch.randint(0, num_steps, (images.shape[0],))

x_t, noise = q_sample_images(images, t, alpha_bars)

print(images.shape)
print(x_t.shape)
print(noise.shape)