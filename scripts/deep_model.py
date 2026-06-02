"""Deep learning model (Approach 3 of 3): fine-tuned MobileNetV3-Small.

We transfer-learn from ImageNet weights because TrashNet is small (~2.5k images).
MobileNetV3-Small is chosen deliberately: it is accurate yet tiny (~2.5M params,
~10 MB), so it runs fast on a CPU-only host like Railway — satisfying the
"inference only, publicly accessible" deployment requirement without a GPU.

Includes a Grad-CAM implementation so the web app can explain *why* a prediction
was made (Selvaraju et al., 2017: https://arxiv.org/abs/1610.02391).

torchvision MobileNetV3:
https://pytorch.org/vision/stable/models/mobilenetv3.html
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from config import (
    CLASS_NAMES,
    DEEP_BATCH_SIZE,
    DEEP_EPOCHS,
    DEEP_LR,
    DEEP_MODEL_PATH,
    DEEP_WEIGHT_DECAY,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    device,
)
from data_utils import get_split, load_image


# --- Transforms --------------------------------------------------------------
def build_transforms(train: bool) -> transforms.Compose:
    """Return the preprocessing/augmentation pipeline for a split."""
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# Inference transform reused by the web app (no augmentation).
INFERENCE_TRANSFORM = build_transforms(train=False)


class TrashDataset(Dataset):
    """Loads images + labels for a split using the persisted split index."""

    def __init__(self, split: str, train: bool):
        self.paths, self.labels = get_split(split)
        self.tf = build_transforms(train=train)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = load_image(self.paths[idx])
        return self.tf(img), int(self.labels[idx])


def build_model(pretrained: bool = True) -> nn.Module:
    """Construct MobileNetV3-Small with a 6-way classification head."""
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    return model


def _class_weights() -> torch.Tensor:
    """Inverse-frequency weights to counter class imbalance (e.g. rare 'trash')."""
    _, y = get_split("train")
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float32)
    weights = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def train_deep(epochs: int = DEEP_EPOCHS) -> Dict:
    """Fine-tune MobileNetV3-Small and persist the best validation checkpoint."""
    dev = device()
    torch.manual_seed(42)
    print(f"[deep] training on device={dev} for {epochs} epochs")

    train_loader = DataLoader(TrashDataset("train", train=True),
                              batch_size=DEEP_BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(TrashDataset("val", train=False),
                            batch_size=DEEP_BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model(pretrained=True).to(dev)
    criterion = nn.CrossEntropyLoss(weight=_class_weights().to(dev))
    optimizer = torch.optim.AdamW(model.parameters(), lr=DEEP_LR, weight_decay=DEEP_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc, best_state, history = 0.0, None, []
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(dev), targets.to(dev)
            optimizer.zero_grad()
            loss = criterion(model(imgs), targets)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
        scheduler.step()
        train_loss = running / len(train_loader.dataset)

        val_acc = _evaluate_accuracy(model, val_loader, dev)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_acc": val_acc})
        print(f"[deep] epoch {epoch + 1}/{epochs} loss={train_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    torch.save({
        "state_dict": best_state,
        "class_names": CLASS_NAMES,
        "arch": "mobilenet_v3_small",
        "image_size": IMAGE_SIZE,
        "best_val_acc": best_val_acc,
        "history": history,
    }, DEEP_MODEL_PATH)
    print(f"[deep] saved best (val_acc={best_val_acc:.4f}) -> {DEEP_MODEL_PATH}")
    return {"best_val_acc": best_val_acc, "history": history}


@torch.no_grad()
def _evaluate_accuracy(model: nn.Module, loader: DataLoader, dev: str) -> float:
    model.eval()
    correct = total = 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(dev), targets.to(dev)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return correct / max(total, 1)


class DeepClassifier:
    """Inference wrapper around the trained MobileNetV3 (used by app + eval)."""

    def __init__(self, model: nn.Module, dev: str):
        self.model = model.eval()
        self.dev = dev

    @staticmethod
    def load() -> "DeepClassifier":
        dev = device()
        ckpt = torch.load(DEEP_MODEL_PATH, map_location=dev)
        model = build_model(pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        model.to(dev)
        return DeepClassifier(model, dev)

    @torch.no_grad()
    def predict_proba_image(self, image: Image.Image) -> np.ndarray:
        x = INFERENCE_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(self.dev)
        logits = self.model(x)
        return torch.softmax(logits, dim=1)[0].cpu().numpy()

    @torch.no_grad()
    def predict_proba_batch(self, images: List[Image.Image]) -> np.ndarray:
        batch = torch.stack([INFERENCE_TRANSFORM(im.convert("RGB")) for im in images]).to(self.dev)
        return torch.softmax(self.model(batch), dim=1).cpu().numpy()

    def grad_cam(self, image: Image.Image, target_class: int | None = None) -> Tuple[np.ndarray, int]:
        """Compute a Grad-CAM heatmap (HxW in [0,1]) for the given image.

        Returns (heatmap, predicted_class).
        """
        target_layer = self.model.features[-1]
        activations: List[torch.Tensor] = []
        gradients: List[torch.Tensor] = []

        def fwd_hook(_m, _i, output):
            activations.append(output)

        def bwd_hook(_m, _gi, grad_output):
            gradients.append(grad_output[0])

        h1 = target_layer.register_forward_hook(fwd_hook)
        h2 = target_layer.register_full_backward_hook(bwd_hook)
        try:
            x = INFERENCE_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(self.dev)
            x.requires_grad_(True)
            logits = self.model(x)
            pred = int(logits.argmax(dim=1).item()) if target_class is None else int(target_class)
            self.model.zero_grad()
            logits[0, pred].backward()

            acts = activations[0].detach()[0]           # (C, h, w)
            grads = gradients[0].detach()[0]            # (C, h, w)
            weights = grads.mean(dim=(1, 2), keepdim=True)
            cam = torch.relu((weights * acts).sum(dim=0))
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            return cam.cpu().numpy(), pred
        finally:
            h1.remove()
            h2.remove()


if __name__ == "__main__":
    train_deep()
