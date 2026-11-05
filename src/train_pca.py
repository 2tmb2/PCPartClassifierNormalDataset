import hashlib
from copy import deepcopy
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestCentroid

from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "laptop_parts"
EPOCHS = 1000
BATCH_SIZE = 128
LEARNING_RATE = 1e-2
OUTPUT_DIR = ROOT / "models"
FEATURE_CACHE_PATH = ROOT / "models" / "laptop_parts_resnet18_features.pth"
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
PCA_COMPONENTS = 128


def make_feature_data():
    preprocessing = [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=WEIGHT_TRANSFORMS.interpolation,
        ),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
    dataset = datasets.ImageFolder(DATA_DIR, transforms.Compose(preprocessing))
    indices = range(len(dataset))
    train_indices, validation_indices = train_test_split(
        indices,
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
            cache["class_names"] == dataset.classes
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


def train_pca(features, labels, train_indices, validation_indices):
    train_features = features[list(train_indices)].numpy()
    train_labels = labels[list(train_indices)].numpy()

    validation_features = features[list(validation_indices)].numpy()
    validation_labels = labels[list(validation_indices)].numpy()

    # Fit PCA using training data only.
    pca = PCA(n_components=PCA_COMPONENTS)
    train_features = pca.fit_transform(train_features)
    validation_features = pca.transform(validation_features)

    print(
        f"PCA explained variance: "
        f"{pca.explained_variance_ratio_.sum():.2%}"
    )

    # Classify using nearest centroid.
    classifier = NearestCentroid()
    classifier.fit(train_features, train_labels)

    train_accuracy = classifier.score(train_features, train_labels)
    validation_accuracy = classifier.score(
        validation_features,
        validation_labels,
    )

    print(f"Training accuracy: {train_accuracy:.2%}")
    print(f"Validation accuracy: {validation_accuracy:.2%}")

    return pca, classifier, validation_accuracy


def checkpoint_path(validation_accuracy):
    filename = (
        f"valacc{validation_accuracy * 100:06.2f}pct_laptop_parts_resnet18_"
        f"epochs{EPOCHS}_batchsize{BATCH_SIZE}_learningrate{LEARNING_RATE}.pth"
    )
    return OUTPUT_DIR / filename

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

    feature_extractor = nn.Sequential(
        *list(model.children())[:-1]
    ).to(device)

    features, labels = load_or_precompute_features(
        dataset, feature_loader, feature_extractor, device
    )

    print(f"Cached features for {len(features)} images.")

    pca, classifier, validation_accuracy = train_pca(
        features,
        labels,
        train_indices,
        validation_indices,
    )

    output_path = OUTPUT_DIR / (
        f"valacc{validation_accuracy * 100:06.2f}pct_"
        f"laptop_parts_resnet18_pca{PCA_COMPONENTS}.pth"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "architecture": "resnet18",
            "classifier": "pca_nearest_centroid",
            "class_names": class_names,
            "image_size": IMAGE_SIZE,
            "image_mean": MEAN,
            "image_std": STD,
            "pca_components": PCA_COMPONENTS,
            "pca": pca,
            "classifier_model": classifier,
            "validation_accuracy": validation_accuracy,
        },
        output_path,
    )

    print(
        f"Validation accuracy: {validation_accuracy:.2%}; "
        f"saved to {output_path}"
    )

if __name__ == "__main__":
    for components in [32, 64, 128, 256, 512]:
        PCA_COMPONENTS = components
        train()