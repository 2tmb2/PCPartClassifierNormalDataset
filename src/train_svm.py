
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "pc_parts"
OUTPUT_DIR = ROOT / "models" / "pc_parts"
GRAPH_DIR = OUTPUT_DIR / "graphs"

FEATURE_CACHE_PATH = OUTPUT_DIR / "pc_parts_resnet18_features.pth"

VALIDATION_FRACTION = 0.2
SEED = 42
NUM_WORKERS = 0
BATCH_SIZE = 128

AUGMENTATION_COPIES = 4
ROTATION_DEGREES = 15

PCA_COMPONENTS = [32, 64, 128, 256, 512]

SVM_KERNEL = "rbf"
SVM_C = 1.0
SVM_GAMMA = "scale"

WEIGHTS = ResNet18_Weights.DEFAULT
WEIGHT_TRANSFORMS = WEIGHTS.transforms()

IMAGE_SIZE = WEIGHT_TRANSFORMS.crop_size[0]
MEAN = WEIGHT_TRANSFORMS.mean
STD = WEIGHT_TRANSFORMS.std


# --------------------------------------------------
# Dataset preparation
# --------------------------------------------------

def make_feature_data():
    preprocessing = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=WEIGHT_TRANSFORMS.interpolation,
        ),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    dataset = datasets.ImageFolder(
        DATA_DIR,
        preprocessing,
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

    augmented_train_count = (
        len(train_indices) * AUGMENTATION_COPIES
    )

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


# --------------------------------------------------
# Feature caching
# --------------------------------------------------

def feature_cache_signature(dataset):
    digest = hashlib.sha256()

    settings = (
        f"{WEIGHTS.name}|{IMAGE_SIZE}|{MEAN}|{STD}|"
        f"augmentation={AUGMENTATION_COPIES}x,"
        f"rotation={ROTATION_DEGREES},"
        "horizontal_flip=0.5,vertical_flip=0.5|"
        f"validation_fraction={VALIDATION_FRACTION}|seed={SEED}|"
    )

    digest.update(settings.encode())

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

    with torch.inference_mode():
        for batch_index, (images, batch_labels) in enumerate(
            loader,
            start=1,
        ):
            images = images.to(device)

            extracted = feature_extractor(images)["features"]

            features.append(extracted.flatten(1).cpu())
            labels.append(batch_labels)

            if batch_index % 50 == 0:
                print(
                    f"Extracted {batch_index * len(images)} images..."
                )

    return torch.cat(features), torch.cat(labels)


def load_or_precompute_features(
    dataset,
    loader,
    feature_extractor,
    device,
):
    signature = feature_cache_signature(dataset)

    if FEATURE_CACHE_PATH.exists():
        cache = torch.load(
            FEATURE_CACHE_PATH,
            map_location="cpu",
            weights_only=True,
        )

        if (
            cache.get("signature") == signature
            and cache.get("class_names") == dataset.classes
        ):
            print(f"Using cached features from {FEATURE_CACHE_PATH}")

            return cache["features"], cache["labels"]

        print("Feature cache is outdated. Rebuilding...")

    print("Precomputing ResNet18 features...")

    features, labels = precompute_features(
        feature_extractor,
        loader,
        device,
    )

    FEATURE_CACHE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_cache = FEATURE_CACHE_PATH.with_name(
        FEATURE_CACHE_PATH.name + ".tmp"
    )

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


# --------------------------------------------------
# PCA + SVM
# --------------------------------------------------

def train_svm(
    features,
    labels,
    train_indices,
    validation_indices,
    components,
):
    train_features = features[list(train_indices)].numpy()
    train_labels = labels[list(train_indices)].numpy()

    validation_features = features[
        list(validation_indices)
    ].numpy()

    validation_labels = labels[
        list(validation_indices)
    ].numpy()

    if components > min(train_features.shape):
        raise ValueError(
            f"Cannot use {components} PCA components. "
            f"Maximum possible: {min(train_features.shape)}"
        )

    # Fit PCA using training data only.
    pca = PCA(
        n_components=components,
        svd_solver="full",
    )

    train_features = pca.fit_transform(train_features)

    validation_features = pca.transform(
        validation_features
    )

    explained_variance = pca.explained_variance_ratio_.sum()

    print(f"PCA explained variance: {explained_variance:.2%}")

    # Train SVM classifier.
    classifier = SVC(
        kernel=SVM_KERNEL,
        C=SVM_C,
        gamma=SVM_GAMMA,
    )

    classifier.fit(
        train_features,
        train_labels,
    )

    train_predictions = classifier.predict(train_features)

    validation_predictions = classifier.predict(
        validation_features
    )

    train_accuracy = np.mean(
        train_predictions == train_labels
    )

    validation_accuracy = np.mean(
        validation_predictions == validation_labels
    )

    print(f"Training accuracy: {train_accuracy:.2%}")
    print(f"Validation accuracy: {validation_accuracy:.2%}")

    return {
        "pca": pca,
        "classifier": classifier,
        "train_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy,
        "explained_variance": explained_variance,
        "validation_predictions": validation_predictions,
        "validation_labels": validation_labels,
    }


# --------------------------------------------------
# Graph generation
# --------------------------------------------------

def plot_pca_variance(pca, components):
    individual = pca.explained_variance_ratio_
    cumulative = np.cumsum(individual)

    component_numbers = np.arange(1, len(individual) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.bar(
        component_numbers,
        individual,
        alpha=0.6,
        label="Individual explained variance",
    )

    ax1.set_xlabel("PCA Component")
    ax1.set_ylabel("Individual Explained Variance")

    ax1.set_title(
        f"PCA Explained Variance ({components} Components)"
    )

    ax2 = ax1.twinx()

    ax2.plot(
        component_numbers,
        cumulative,
        linewidth=2,
        label="Cumulative explained variance",
    )

    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_ylim(0, 1.05)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="center right",
    )

    fig.tight_layout()

    path = GRAPH_DIR / f"pca_variance_{components}.png"

    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(f"Saved {path}")


def plot_confusion_matrix(
    predictions,
    actual,
    class_names,
    components,
):
    matrix = confusion_matrix(
        actual,
        predictions,
        labels=np.arange(len(class_names)),
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(matrix, interpolation="nearest")

    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted Class",
        ylabel="Actual Class",
        title=f"SVM Confusion Matrix ({components} PCA Components)",
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    threshold = matrix.max() / 2 if matrix.size else 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )

    fig.tight_layout()

    path = GRAPH_DIR / f"confusion_matrix_{components}.png"

    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(f"Saved {path}")

    return matrix


def plot_class_accuracy(matrix, class_names, components):
    correct = np.diag(matrix)
    totals = matrix.sum(axis=1)

    accuracy = np.divide(
        correct,
        totals,
        out=np.full(len(class_names), np.nan),
        where=totals != 0,
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        class_names,
        accuracy * 100,
    )

    ax.set(
        xlabel="Class",
        ylabel="Validation Accuracy (%)",
        title=f"Per-Class SVM Accuracy ({components} PCA Components)",
        ylim=(0, 105),
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
    )

    fig.tight_layout()

    path = GRAPH_DIR / f"class_accuracy_{components}.png"

    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(f"Saved {path}")


def plot_comparison(results):
    components = [r["components"] for r in results]

    train_accuracy = [
        r["train_accuracy"] * 100 for r in results
    ]

    validation_accuracy = [
        r["validation_accuracy"] * 100 for r in results
    ]

    explained_variance = [
        r["explained_variance"] * 100 for r in results
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        components,
        train_accuracy,
        marker="o",
        label="Training Accuracy",
    )

    ax.plot(
        components,
        validation_accuracy,
        marker="o",
        label="Validation Accuracy",
    )

    ax.plot(
        components,
        explained_variance,
        marker="o",
        linestyle="--",
        label="Explained Variance",
    )

    ax.set(
        xlabel="Number of PCA Components",
        ylabel="Percentage (%)",
        title="PCA + SVM Component Comparison",
        ylim=(0, 105),
        xticks=components,
    )

    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    path = GRAPH_DIR / "svm_pca_comparison.png"

    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(f"Saved {path}")


# --------------------------------------------------
# Model training and evaluation
# --------------------------------------------------

def train(components, feature_data):
    print(f"\n{'=' * 50}")
    print(f"Testing SVM with {components} PCA components")
    print(f"{'=' * 50}")

    features, labels, train_indices, validation_indices, dataset = (
        feature_data
    )

    result = train_svm(
        features,
        labels,
        train_indices,
        validation_indices,
        components,
    )

    pca = result["pca"]
    classifier = result["classifier"]

    # Generate graphs.
    plot_pca_variance(pca, components)

    matrix = plot_confusion_matrix(
        result["validation_predictions"],
        result["validation_labels"],
        dataset.classes,
        components,
    )

    plot_class_accuracy(
        matrix,
        dataset.classes,
        components,
    )

    # Save trained PCA and SVM.
    output_path = OUTPUT_DIR / (
        f"valacc{result['validation_accuracy'] * 100:06.2f}pct_"
        f"resnet18_svm_pca{components}.pth"
    )

    torch.save(
        {
            "architecture": "resnet18",
            "classifier": "pca_svm",
            "svm_kernel": SVM_KERNEL,
            "svm_c": SVM_C,
            "svm_gamma": SVM_GAMMA,
            "class_names": dataset.classes,
            "image_size": IMAGE_SIZE,
            "image_mean": MEAN,
            "image_std": STD,
            "pca_components": components,
            "pca": pca,
            "classifier_model": classifier,
            "validation_accuracy": result["validation_accuracy"],
        },
        output_path,
    )

    print(f"Saved model to {output_path}")

    return {
        "components": components,
        "train_accuracy": result["train_accuracy"],
        "validation_accuracy": result["validation_accuracy"],
        "explained_variance": result["explained_variance"],
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    torch.manual_seed(SEED)

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    # Prepare dataset.
    feature_loader, train_indices, validation_indices, dataset = (
        make_feature_data()
    )

    # Load pretrained ResNet18 feature extractor.
    model = models.resnet18(weights=WEIGHTS)

    feature_extractor = create_feature_extractor(
        model,
        return_nodes={"avgpool": "features"},
    ).to(device)

    # Extract or load cached features.
    features, labels = load_or_precompute_features(
        dataset,
        feature_loader,
        feature_extractor,
        device,
    )

    print(f"Cached features for {len(features)} images.")

    feature_data = (
        features,
        labels,
        train_indices,
        validation_indices,
        dataset,
    )

    # Evaluate different PCA configurations.
    results = []

    for components in PCA_COMPONENTS:
        results.append(
            train(components, feature_data)
        )

    # Generate comparison graph.
    plot_comparison(results)

    print("\nFinal Results")
    print("-" * 70)

    for result in results:
        print(
            f"PCA: {result['components']:3d} | "
            f"Train: {result['train_accuracy']:.2%} | "
            f"Validation: {result['validation_accuracy']:.2%} | "
            f"Explained Variance: {result['explained_variance']:.2%}"
        )

    print(f"\nGraphs saved to {GRAPH_DIR}")


if __name__ == "__main__":
    main()
