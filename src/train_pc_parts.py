import hashlib
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "pc_parts"
EPOCHS = 500
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
OUTPUT_DIR = ROOT / "models" / "pc_parts"
GRAPH_DIR = OUTPUT_DIR / "graphs"
FEATURE_CACHE_PATH = OUTPUT_DIR / "pc_parts_resnet18_features.pth"
VALIDATION_FRACTION = 0.2
SEED = 42
NUM_WORKERS = 0
AUGMENTATION_COPIES = 4
ROTATION_DEGREES = 15
WEIGHTS = ResNet18_Weights.DEFAULT
WEIGHT_TRANSFORMS = WEIGHTS.transforms()
IMAGE_SIZE = WEIGHT_TRANSFORMS.crop_size[0]
MEAN = WEIGHT_TRANSFORMS.mean
STD = WEIGHT_TRANSFORMS.std


def make_feature_data():
    preprocessing = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=WEIGHT_TRANSFORMS.interpolation,
        ),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    dataset = datasets.ImageFolder(DATA_DIR, preprocessing)
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
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=WEIGHT_TRANSFORMS.interpolation,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(ROTATION_DEGREES),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]),
    )
    augmented_train_dataset = ConcatDataset([
        Subset(augmented_dataset, train_indices)
        for _ in range(AUGMENTATION_COPIES)
    ])
    feature_dataset = ConcatDataset([
        augmented_train_dataset,
        Subset(dataset, validation_indices),
    ])
    augmented_train_count = len(train_indices) * AUGMENTATION_COPIES
    feature_loader = DataLoader(
        feature_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    return (
        feature_loader,
        range(augmented_train_count),
        range(augmented_train_count, len(feature_dataset)),
        dataset,
    )


def feature_cache_signature(dataset):
    digest = hashlib.sha256()
    digest.update(
        f"{WEIGHTS.name}|{IMAGE_SIZE}|{MEAN}|{STD}|"
        f"augmentation={AUGMENTATION_COPIES}x,rotation={ROTATION_DEGREES},"
        "horizontal_flip=0.5,vertical_flip=0.5|"
        f"validation_fraction={VALIDATION_FRACTION}|seed={SEED}|".encode()
    )
    for path, target in dataset.samples:
        image_path = Path(path)
        stat = image_path.stat()
        digest.update(
            f"{image_path.relative_to(DATA_DIR)}|{target}|"
            f"{stat.st_size}|{stat.st_mtime_ns}\n".encode()
        )
    return digest.hexdigest()


def precompute_features(feature_extractor, loader, device):
    features = []
    labels = []
    feature_extractor.eval()
    with torch.no_grad():
        for batch_index, (images, batch_labels) in enumerate(loader, start=1):
            images = images.to(device)
            features.append(feature_extractor(images).flatten(1).cpu())
            labels.append(batch_labels)
            if batch_index % 50 == 0:
                print(f"Extracted {batch_index * len(images)} images...")
    return torch.cat(features), torch.cat(labels)


def load_or_precompute_features(dataset, loader, feature_extractor, device):
    signature = feature_cache_signature(dataset)
    if FEATURE_CACHE_PATH.exists():
        cache = torch.load(FEATURE_CACHE_PATH, map_location="cpu", weights_only=True)
        if (
            cache["signature"] == signature
            and cache["class_names"] == dataset.classes
        ):
            print(f"Using cached features from {FEATURE_CACHE_PATH}")
            return cache["features"], cache["labels"]
        print("Feature cache does not match the dataset or preprocessing; rebuilding it.")

    print("Precomputing frozen ResNet features...")
    features, labels = precompute_features(feature_extractor, loader, device)
    FEATURE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_cache = FEATURE_CACHE_PATH.with_name(FEATURE_CACHE_PATH.name + ".tmp")
    torch.save(
        {
            "signature": signature,
            "class_names": dataset.classes,
            "features": features,
            "labels": labels,
        },
        temporary_cache,
    )
    temporary_cache.replace(FEATURE_CACHE_PATH)
    print(f"Saved feature cache to {FEATURE_CACHE_PATH}")
    return features, labels


def make_feature_loaders(features, labels, train_indices, validation_indices):
    dataset = TensorDataset(features, labels)
    return (
        DataLoader(
            Subset(dataset, train_indices),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
        ),
        DataLoader(
            Subset(dataset, validation_indices),
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        ),
    )


def run_epoch(classifier, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    classifier.train(training)
    total_loss = total_correct = total = 0

    with torch.set_grad_enabled(training):
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            predictions = classifier(features)
            loss = loss_fn(predictions, labels)
            if training:
                loss.backward()
                optimizer.step()

            count = labels.size(0)
            total_loss += loss.item() * count
            total_correct += (predictions.argmax(1) == labels).sum().item()
            total += count
    return total_loss / total, total_correct / total


def checkpoint_path(validation_accuracy):
    filename = (
        f"valacc{validation_accuracy * 100:06.2f}pct_pc_parts_resnet18_"
        f"epochs{EPOCHS}_batchsize{BATCH_SIZE}_learningrate{LEARNING_RATE}.pth"
    )
    return OUTPUT_DIR / filename


def save_loss_plot(train_losses, validation_losses, validation_accuracy):
    figure, axis = plt.subplots()
    epochs = range(1, len(train_losses) + 1)
    axis.plot(epochs, train_losses, label="Training loss")
    axis.plot(epochs, validation_losses, label="Validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("PC Parts Training and Validation Loss")
    axis.legend()
    figure.tight_layout()

    graph_path = GRAPH_DIR / checkpoint_path(validation_accuracy).with_suffix(".png").name
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(graph_path)
    plt.close(figure)
    return graph_path


def train() -> None:
    torch.manual_seed(SEED)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    feature_loader, train_indices, validation_indices, dataset = make_feature_data()
    class_names = dataset.classes

    model = models.resnet18(weights=WEIGHTS)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.to(device)

    feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(device)
    features, labels = load_or_precompute_features(
        dataset, feature_loader, feature_extractor, device
    )
    train_loader, validation_loader = make_feature_loaders(
        features, labels, train_indices, validation_indices
    )
    print(f"Dataset: {len(dataset)} images | Classes: {len(class_names)}")

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    best_accuracy = -1.0
    best_state = None
    train_losses = []
    validation_losses = []

    print(f"Device: {device} | Classes: {class_names}")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(
            model.fc, train_loader, loss_function, device, optimizer
        )
        validation_loss, validation_accuracy = run_epoch(
            model.fc, validation_loader, loss_function, device
        )
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        print(
            f"Epoch {epoch}/{EPOCHS}: "
            f"train loss {train_loss:.4f}, accuracy {train_accuracy:.2%}; "
            f"validation loss {validation_loss:.4f}, "
            f"accuracy {validation_accuracy:.2%}"
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = deepcopy(model.fc.state_dict())

    model.fc.load_state_dict(best_state)
    output_path = checkpoint_path(best_accuracy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "resnet18",
            "class_names": class_names,
            "image_size": IMAGE_SIZE,
            "image_mean": MEAN,
            "image_std": STD,
            "validation_accuracy": best_accuracy,
            "best_epoch": best_epoch,
            "model_state_dict": model.state_dict(),
        },
        output_path,
    )
    graph_path = save_loss_plot(train_losses, validation_losses, best_accuracy)
    print(f"Best validation accuracy: {best_accuracy:.2%} at epoch {best_epoch}")
    print(f"Saved model to {output_path}")
    print(f"Saved loss graph to {graph_path}")


if __name__ == "__main__":
    train()
