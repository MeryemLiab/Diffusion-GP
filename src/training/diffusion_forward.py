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
