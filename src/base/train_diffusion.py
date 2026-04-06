import torch
import torch.nn as nn
import torch.optim as optim

from src.diffusion_utils import prepare_noise_schedule, q_sample


def train_diffusion(model, x, y_clean, num_steps=100, epochs=1000, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    x = x.to(device).float()
    y_clean = y_clean.to(device).float()

    _, _, alpha_bars = prepare_noise_schedule(num_steps)
    alpha_bars = alpha_bars.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []

    for epoch in range(epochs):
        model.train()

        t = torch.randint(0, num_steps, (len(x),), device=device)
        y_t, noise = q_sample(y_clean, t, alpha_bars)
        t_normalized = t.float() / (num_steps - 1)
        pred_noise = model(x, y_t.squeeze(1), t_normalized)

        loss = criterion(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss = {loss.item():.6f}")

    return model, losses, alpha_bars