# %%
import torch 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloader(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return dataloader


# %%

dataloader = get_mnist_dataloader(batch_size=64)
images, labels = next(iter(dataloader))

print(images.shape)
print(labels.shape)


# %%
import matplotlib.pyplot as plt

img = images[18].squeeze().numpy()

plt.imshow(img, cmap="gray")
plt.title(f"Label: {labels[18]}")
plt.show()
# %%
