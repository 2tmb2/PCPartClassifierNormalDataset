"""
Evaluate a saved PC parts classifier on the augmented train/validation dataset.

Set MODEL_PATH to a checkpoint path relative to models/pc_parts, then run:
    python src/confusion.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "pc_parts"
MODEL_DIR = ROOT / "models" / "pc_parts"
GRAPH_DIR = MODEL_DIR / "graphs"
MODEL_PATH = (
    "valacc087.37pct_pc_parts_resnet18_epochs50_maxepochs50_bestepoch40_"
    "completed_batchsize32_learningrate0.0001.pth"
)
VALIDATION_FRACTION = 0.2
SEED = 41
NUM_WORKERS = 0
AUGMENTATION_COPIES = 4
ROTATION_DEGREES = 15
WEIGHT_TRANSFORMS = ResNet18_Weights.DEFAULT.transforms()


def make_evaluation_data(checkpoint: dict) -> tuple[DataLoader, list[str]]:
    preprocessing = transforms.Compose([
        transforms.Resize(
            (checkpoint["image_size"], checkpoint["image_size"]),
            interpolation=WEIGHT_TRANSFORMS.interpolation,
        ),
        transforms.ToTensor(),
        transforms.Normalize(checkpoint["image_mean"], checkpoint["image_std"]),
    ])
    dataset = datasets.ImageFolder(DATA_DIR, preprocessing)
    if dataset.classes != checkpoint["class_names"]:
        raise ValueError(
            "Dataset classes do not match the checkpoint class names: "
            f"{dataset.classes} != {checkpoint['class_names']}"
        )

    train_indices, validation_indices = train_test_split(
        range(len(dataset)),
        test_size=VALIDATION_FRACTION,
        random_state=SEED,
        stratify=dataset.targets,
    )
    augmented_dataset = datasets.ImageFolder(
        DATA_DIR,
        transforms.Compose([
            transforms.Resize(
                (checkpoint["image_size"], checkpoint["image_size"]),
                interpolation=WEIGHT_TRANSFORMS.interpolation,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(ROTATION_DEGREES),
            transforms.ToTensor(),
            transforms.Normalize(checkpoint["image_mean"], checkpoint["image_std"]),
        ]),
    )
    augmented_train_dataset = ConcatDataset([
        Subset(augmented_dataset, train_indices)
        for _ in range(AUGMENTATION_COPIES)
    ])
    evaluation_dataset = ConcatDataset([
        augmented_train_dataset,
        Subset(dataset, validation_indices),
    ])
    loader = DataLoader(
        evaluation_dataset,
        batch_size=32,
        num_workers=NUM_WORKERS,
    )
    return loader, dataset.classes


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "resnet18":
        raise ValueError("This evaluator supports ResNet-18 checkpoints only.")

    class_names = checkpoint.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("Checkpoint is missing a valid class_names list.")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device,
) -> tuple[float, float, np.ndarray]:
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            count = labels.size(0)
            total_loss += loss_function(logits, labels).item() * count
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total += count
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    if total == 0:
        raise ValueError("The evaluation dataset is empty.")
    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=range(len(class_names)),
    )
    return total_loss / total, total_correct / total, matrix


def print_metrics(matrix: np.ndarray, class_names: list[str]) -> None:
    print("\nPer-class metrics:")
    print(
        f"{'Class':<16} {'Precision':>10} {'Recall':>10} {'FPR':>10} {'F1':>10}"
    )
    precision_values = []
    recall_values = []
    fpr_values = []
    f1_values = []
    for index, class_name in enumerate(class_names):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum() - true_positive)
        false_negative = int(matrix[index, :].sum() - true_positive)
        true_negative = int(
            matrix.sum() - true_positive - false_positive - false_negative
        )
        precision = _ratio(true_positive, true_positive + false_positive)
        recall = _ratio(true_positive, true_positive + false_negative)
        false_positive_rate = _ratio(false_positive, false_positive + true_negative)
        f1 = _ratio(2 * precision * recall, precision + recall)
        print(
            f"{class_name:<16} {precision:>10.4f} {recall:>10.4f} "
            f"{false_positive_rate:>10.4f} {f1:>10.4f}"
        )
        # save values for later reference
        precision_values.append(precision)
        recall_values.append(recall)
        fpr_values.append(false_positive_rate)
        f1_values.append(f1)

    print(
        f"\nMacro average:\n  Precision: {np.mean(precision_values):.4f}\n  "
        f"Recall={np.mean(recall_values):.4f}\n  FPR: {np.mean(fpr_values):.4f}\n  "
        f"F1={np.mean(f1_values):.4f}"
    )


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def save_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    checkpoint_path: Path,
) -> Path:
    figure, axis = plt.subplots(
        figsize=(max(8, len(class_names)), max(6, len(class_names) * 0.75))
    )
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=f"Confusion Matrix: {checkpoint_path.stem}",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    graph_path = GRAPH_DIR / f"{checkpoint_path.stem}_confusion_matrix.png"
    figure.savefig(graph_path)
    plt.close(figure)
    return graph_path


def main() -> None:
    model_relative_path = Path(MODEL_PATH)
    if model_relative_path.is_absolute():
        raise ValueError("MODEL_PATH must be relative to models/pc_parts.")
    checkpoint_path = MODEL_DIR / model_relative_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model, checkpoint = load_model(checkpoint_path, device)
    loader, class_names = make_evaluation_data(checkpoint)
    print(f"Model: {checkpoint_path}")
    print(f"Device: {device} | Evaluation images: {len(loader.dataset)}")
    loss, accuracy, matrix = evaluate(model, loader, class_names, device)
    print(f"Loss: {loss:.4f} | Accuracy: {accuracy:.2%}")
    print_metrics(matrix, class_names)
    graph_path = save_confusion_matrix(matrix, class_names, checkpoint_path)
    print(f"Saved confusion matrix to {graph_path}")


if __name__ == "__main__":
    main()
