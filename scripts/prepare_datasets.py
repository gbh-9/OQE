"""Download the four public torchvision datasets into the expected layout."""

from pathlib import Path

from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, KMNIST

ROOT = Path(__file__).resolve().parents[1] / "data"

for name, cls in (
    ("fashion_mnist", FashionMNIST),
    ("kmnist", KMNIST),
    ("cifar10", CIFAR10),
    ("cifar100", CIFAR100),
):
    for train in (True, False):
        cls(ROOT / name, train=train, download=True)
    print(f"prepared {name}")
