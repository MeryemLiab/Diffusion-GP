import torch


def linear_beta_schedule(num_steps, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, num_steps)


def prepare_noise_schedule(num_steps, beta_start=1e-4, beta_end=2e-2):
    betas = linear_beta_schedule(num_steps, beta_start, beta_end)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    return betas, alphas, alpha_bars


def q_sample(y_clean, t, alpha_bars):
    if y_clean.dim() == 1:
        y_clean = y_clean.unsqueeze(1)

    noise = torch.randn_like(y_clean)
    alpha_bar_t = alpha_bars[t].unsqueeze(1)

    y_t = torch.sqrt(alpha_bar_t) * y_clean + torch.sqrt(1.0 - alpha_bar_t) * noise

    return y_t, noise