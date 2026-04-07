import torch
import torch.nn as nn
import torch.optim as optim


def train_mnist_diffusion(model, dataloader, q_sample_images, alpha_bars, num_steps, epochs=5, lr=1e-3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    alpha_bars = alpha_bars.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for images, _ in dataloader:
            images = images.to(device)

            batch_size = images.shape[0]

            t = torch.randint(0, num_steps, (batch_size,), device=device)

            x_t, noise = q_sample_images(images, t, alpha_bars)

            t_normalized = t.float() / (num_steps - 1)

            pred_noise = model(x_t, t_normalized)

            # 5. loss
            loss = criterion(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

    return model, losses