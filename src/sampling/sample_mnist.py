import torch

def sample_images(model, alpha_bars, num_steps, shape, device):
    model.eval()

    with torch.no_grad():
        x_t = torch.randn(shape).to(device)

        for t in reversed(range(num_steps)):
            t_tensor = torch.full((shape[0],), t, device=device)
            t_norm = t_tensor.float() / (num_steps - 1)

            pred_noise = model(x_t, t_norm)

            alpha_bar_t = alpha_bars[t].to(device)

            x_t = (x_t - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)

        return x_t