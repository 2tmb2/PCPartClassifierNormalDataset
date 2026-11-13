import hashlib
import json
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from torchvision.transforms import functional as TF


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "pc_parts"
OUTPUT_DIR = ROOT / "models" / "pc_parts"
GRAPH_DIR = OUTPUT_DIR / "graphs"

CACHE_PATH = OUTPUT_DIR / "pc_parts_resnet18_features.pth"

PCA_COMPONENTS = [32, 64, 128, 256, 512]

TEST_SIZE = 0.20
RANDOM_STATE = 42

BATCH_SIZE = 128
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSIFIERS = ["centroid", "svm"]


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Image preprocessing and augmentation
# ============================================================

weights = ResNet18_Weights.DEFAULT

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def get_training_images(image):
    """
    Return the original image and four deterministic augmentations:
        0: Original
        1: Horizontal flip
        2: Vertical flip
        3: +15 degree rotation
        4: -15 degree rotation
    """

    return [
        image,
        TF.hflip(image),
        TF.vflip(image),
        TF.rotate(image, 15),
        TF.rotate(image, -15),
    ]


class ImageFeatureDataset(Dataset):

    def __init__(self, records):
        """
        Each record contains:
            (image_path, label, augmentation_id)

        augmentation_id:
            None = validation image
            0-4 = training image variant
        """

        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label, augmentation_id = self.records[index]

        image = load_image(path)

        if augmentation_id is not None:
            image = get_training_images(image)[augmentation_id]

        image = preprocess(image)

        return image, label


# ============================================================
# ResNet18 feature extractor
# ============================================================

def create_feature_extractor():
    model = models.resnet18(weights=weights)

    # Remove the final classification layer.
    model.fc = nn.Identity()

    model.eval()
    model.to(DEVICE)

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model


# ============================================================
# Dataset splitting
# ============================================================

def split_dataset(dataset):
    labels = np.array(dataset.targets)
    indices = np.arange(len(dataset))

    train_indices, validation_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    return train_indices, validation_indices


# ============================================================
# Feature extraction records
# ============================================================

def make_feature_records(dataset, train_indices, validation_indices):
    train_records = []
    validation_records = []

    for index in train_indices:
        path, label = dataset.samples[index]

        # Original plus four augmented versions.
        for augmentation_id in range(5):
            train_records.append(
                (path, label, augmentation_id)
            )

    for index in validation_indices:
        path, label = dataset.samples[index]

        # Validation images are not augmented.
        validation_records.append(
            (path, label, None)
        )

    return train_records, validation_records


# ============================================================
# Feature cache
# ============================================================

def feature_cache_signature(dataset, train_indices, validation_indices):
    """
    Generate a signature that changes when the dataset or
    train/validation split changes.
    """

    signature_data = {
        "classes": dataset.classes,
        "train_indices": train_indices.tolist(),
        "validation_indices": validation_indices.tolist(),
        "weights": str(weights),
        "augmentation_count": 5,
        "augmentations": [
            "original",
            "horizontal_flip",
            "vertical_flip",
            "rotate_15",
            "rotate_minus_15",
        ],
        "image_size": 224,
    }

    file_information = []

    for path, _ in dataset.samples:
        path = Path(path)

        stat = path.stat()

        file_information.append({
            "path": str(path.relative_to(DATA_DIR)),
            "size": stat.st_size,
            "modified": stat.st_mtime_ns,
        })

    signature_data["files"] = file_information

    encoded = json.dumps(
        signature_data,
        sort_keys=True,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def extract_features(model, records):
    dataset = ImageFeatureDataset(records)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    all_features = []
    all_labels = []

    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)

            features = model(images)

            all_features.append(
                features.cpu()
            )

            all_labels.append(labels)

    features = torch.cat(all_features).numpy()
    labels = torch.cat(all_labels).numpy()

    return features, labels


def load_or_precompute_features(
    dataset,
    train_indices,
    validation_indices,
):
    signature = feature_cache_signature(
        dataset,
        train_indices,
        validation_indices,
    )

    if CACHE_PATH.exists():
        print("Checking feature cache...")

        try:
            cache = torch.load(
                CACHE_PATH,
                map_location="cpu",
                weights_only=True,
            )

            if (
                cache["signature"] == signature
                and cache["class_names"] == dataset.classes
            ):
                print("Using cached ResNet18 features.")

                return (
                    cache["train_features"].numpy(),
                    cache["train_labels"].numpy(),
                    cache["validation_features"].numpy(),
                    cache["validation_labels"].numpy(),
                )

        except (KeyError, RuntimeError, ValueError, TypeError):
            print("Feature cache is invalid. Recomputing.")

    print("Extracting ResNet18 features...")

    train_records, validation_records = make_feature_records(
        dataset,
        train_indices,
        validation_indices,
    )

    model = create_feature_extractor()

    train_features, train_labels = extract_features(
        model,
        train_records,
    )

    validation_features, validation_labels = extract_features(
        model,
        validation_records,
    )

    cache = {
        "signature": signature,
        "class_names": dataset.classes,
        "train_features": torch.tensor(
            train_features,
            dtype=torch.float32,
        ),
        "train_labels": torch.tensor(
            train_labels,
            dtype=torch.long,
        ),
        "validation_features": torch.tensor(
            validation_features,
            dtype=torch.float32,
        ),
        "validation_labels": torch.tensor(
            validation_labels,
            dtype=torch.long,
        ),
    }

    torch.save(cache, CACHE_PATH)

    print(f"Saved feature cache to {CACHE_PATH}")

    return (
        train_features,
        train_labels,
        validation_features,
        validation_labels,
    )


# ============================================================
# Probability estimation for Nearest Centroid
# ============================================================

def centroid_probabilities(classifier, features):
    """
    Convert distances to NearestCentroid centroids into
    normalized probabilities using softmax.
    """

    # Calculate the Euclidean distance from each sample
    # to each class centroid.
    distances = np.linalg.norm(
        features[:, np.newaxis, :] -
        classifier.centroids_[np.newaxis, :, :],
        axis=2,
    )

    # Convert distances into scores.
    scores = -distances

    # Numerically stable softmax.
    scores -= np.max(scores, axis=1, keepdims=True)

    probabilities = np.exp(scores)

    probabilities /= np.sum(
        probabilities,
        axis=1,
        keepdims=True,
    )

    return probabilities


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    labels,
    predictions,
    probabilities,
    class_ids,
):
    accuracy = accuracy_score(
        labels,
        predictions,
    )

    error_rate = 1.0 - accuracy

    precision = precision_score(
        labels,
        predictions,
        labels=class_ids,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        labels=class_ids,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        labels=class_ids,
        average="macro",
        zero_division=0,
    )

    cce = log_loss(
        labels,
        probabilities,
        labels=class_ids,
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=class_ids,
    )

    false_positive_rates = []

    for index in range(len(class_ids)):
        true_positive = matrix[index, index]

        false_positive = (
            np.sum(matrix[:, index]) - true_positive
        )

        false_negative = (
            np.sum(matrix[index, :]) - true_positive
        )

        true_negative = (
            np.sum(matrix) -
            true_positive -
            false_positive -
            false_negative
        )

        denominator = false_positive + true_negative

        if denominator == 0:
            fpr = 0.0
        else:
            fpr = false_positive / denominator

        false_positive_rates.append(fpr)

    macro_fpr = np.mean(false_positive_rates)

    return {
        "cce": cce,
        "accuracy": accuracy,
        "error_rate": error_rate,
        "precision": precision,
        "recall": recall,
        "fpr": macro_fpr,
        "f1": f1,
    }


# ============================================================
# PCA and classifier training
# ============================================================

def train_classifier(
    classifier_type,
    train_features,
    train_labels,
    validation_features,
    validation_labels,
    components,
):
    """
    Fit PCA using training features only.

    Train either NearestCentroid or RBF SVM and evaluate
    against the validation features.
    """

    max_components = min(
        train_features.shape[0] - 1,
        train_features.shape[1],
    )

    actual_components = min(
        components,
        max_components,
    )

    if actual_components < 1:
        raise ValueError(
            "Not enough training samples to perform PCA."
        )

    pca = PCA(
        n_components=actual_components,
        random_state=RANDOM_STATE,
    )

    reduced_train = pca.fit_transform(
        train_features
    )

    reduced_validation = pca.transform(
        validation_features
    )

    if classifier_type == "centroid":

        classifier = NearestCentroid(
            metric="euclidean",
        )

        classifier.fit(
            reduced_train,
            train_labels,
        )

        train_predictions = classifier.predict(
            reduced_train
        )

        validation_predictions = classifier.predict(
            reduced_validation
        )

        probabilities = centroid_probabilities(
            classifier,
            reduced_validation,
        )

        train_probabilities = centroid_probabilities(
            classifier,
            reduced_train,
        )

    elif classifier_type == "svm":

        classifier = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=RANDOM_STATE,
        )

        classifier.fit(
            reduced_train,
            train_labels,
        )

        train_predictions = classifier.predict(
            reduced_train
        )

        validation_predictions = classifier.predict(
            reduced_validation
        )

        probabilities = classifier.predict_proba(
            reduced_validation
        )

        train_probabilities = classifier.predict_proba(
            reduced_train
        )

    else:
        raise ValueError(
            f"Unknown classifier: {classifier_type}"
        )

    class_ids = np.sort(
        np.unique(train_labels)
    )

    train_metrics = calculate_metrics(
        train_labels,
        train_predictions,
        train_probabilities,
        class_ids,
    )

    validation_metrics = calculate_metrics(
        validation_labels,
        validation_predictions,
        probabilities,
        class_ids,
    )

    return {
        "classifier": classifier_type,
        "requested_components": components,
        "components": actual_components,
        "pca": pca,
        "model": classifier,
        "explained_variance": np.sum(
            pca.explained_variance_ratio_
        ),
        "train_metrics": train_metrics,
        "metrics": validation_metrics,
        "validation_labels": validation_labels,
        "validation_predictions": validation_predictions,
        "validation_probabilities": probabilities,
        "class_ids": class_ids,
    }


# ============================================================
# Plotting: PCA comparison
# ============================================================

def plot_pca_comparison(results, classifier_type):
    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "fpr",
    ]

    components = [
        result["components"]
        for result in results
    ]

    plt.figure(figsize=(11, 6))

    for metric in metrics:
        values = [
            result["metrics"][metric]
            for result in results
        ]

        plt.plot(
            components,
            values,
            marker="o",
            label=metric.upper(),
        )

    plt.xlabel("PCA Components")
    plt.ylabel("Metric Value")
    plt.title(
        f"{classifier_type.title()} - Validation Metrics"
    )

    plt.xticks(components)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / f"{classifier_type}_pca_comparison.png",
        dpi=300,
    )

    plt.close()


def plot_loss_comparison(results, classifier_type):
    components = [
        result["components"]
        for result in results
    ]

    cce = [
        result["metrics"]["cce"]
        for result in results
    ]

    error_rate = [
        result["metrics"]["error_rate"]
        for result in results
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    axes[0].plot(
        components,
        cce,
        marker="o",
    )

    axes[0].set_title("Categorical Cross-Entropy")
    axes[0].set_xlabel("PCA Components")
    axes[0].set_ylabel("CCE")
    axes[0].set_xticks(components)
    axes[0].grid(True)

    axes[1].plot(
        components,
        error_rate,
        marker="o",
    )

    axes[1].set_title("Error Rate")
    axes[1].set_xlabel("PCA Components")
    axes[1].set_ylabel("Error Rate")
    axes[1].set_xticks(components)
    axes[1].grid(True)

    fig.suptitle(
        f"{classifier_type.title()} - Loss and Error"
    )

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / f"{classifier_type}_loss_comparison.png",
        dpi=300,
    )

    plt.close()


# ============================================================
# Plotting: confusion matrix
# ============================================================

def plot_confusion_matrix(result, class_names):
    labels = result["validation_labels"]
    predictions = result["validation_predictions"]

    class_ids = result["class_ids"]

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=class_ids,
    )

    # Normalize each row to show the proportion of each
    # actual class predicted as each class.
    row_sums = matrix.sum(axis=1, keepdims=True)

    normalized = np.divide(
        matrix.astype(float),
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    )

    plt.figure(figsize=(10, 8))

    plt.imshow(
        normalized,
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=1,
    )

    plt.title(
        f"{result['classifier'].title()} Confusion Matrix "
        f"(PCA {result['components']})"
    )

    plt.colorbar()

    ticks = np.arange(len(class_names))

    plt.xticks(
        ticks,
        class_names,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        ticks,
        class_names,
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    for row in range(normalized.shape[0]):
        for column in range(normalized.shape[1]):
            value = normalized[row, column]

            plt.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
            )

    plt.tight_layout()

    filename = (
        f"{result['classifier']}_"
        f"confusion_matrix_pca{result['components']}.png"
    )

    plt.savefig(
        GRAPH_DIR / filename,
        dpi=300,
    )

    plt.close()


# ============================================================
# Plotting: per-class accuracy
# ============================================================

def plot_class_accuracy(result, class_names):
    labels = result["validation_labels"]
    predictions = result["validation_predictions"]

    accuracies = []

    for class_id in range(len(class_names)):
        mask = labels == class_id

        if np.sum(mask) == 0:
            accuracies.append(0.0)
        else:
            accuracies.append(
                accuracy_score(
                    labels[mask],
                    predictions[mask],
                )
            )

    plt.figure(figsize=(12, 6))

    bars = plt.bar(
        class_names,
        accuracies,
    )

    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("Class")
    plt.title(
        f"{result['classifier'].title()} Per-Class Accuracy "
        f"(PCA {result['components']})"
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    for bar, accuracy in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{accuracy:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    filename = (
        f"{result['classifier']}_"
        f"class_accuracy_pca{result['components']}.png"
    )

    plt.savefig(
        GRAPH_DIR / filename,
        dpi=300,
    )

    plt.close()


# ============================================================
# Plotting: best model comparison
# ============================================================

def plot_best_model_comparison(best_models):
    classifier_names = [
        name.title()
        for name in best_models
    ]

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "fpr",
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(
        figsize=(12, 6),
    )

    for index, (classifier_type, result) in enumerate(
        best_models.items()
    ):
        values = [
            result["metrics"][metric]
            for metric in metrics
        ]

        ax.bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=(
                f"{classifier_type.title()} "
                f"(PCA {result['components']})"
            ),
        )

    ax.set_ylabel("Metric Value")
    ax.set_title("Best Model Metric Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [metric.upper() for metric in metrics]
    )

    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "best_model_comparison.png",
        dpi=300,
    )

    plt.close()

    # Compare CCE and error rate separately.
    loss_metrics = ["cce", "error_rate"]

    x = np.arange(len(loss_metrics))

    fig, ax = plt.subplots(
        figsize=(9, 5),
    )

    for index, (classifier_type, result) in enumerate(
        best_models.items()
    ):
        values = [
            result["metrics"][metric]
            for metric in loss_metrics
        ]

        ax.bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=(
                f"{classifier_type.title()} "
                f"(PCA {result['components']})"
            ),
        )

    ax.set_ylabel("Value")
    ax.set_title("Best Model CCE and Error Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["CCE", "Error Rate"]
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "best_model_loss_comparison.png",
        dpi=300,
    )

    plt.close()


# ============================================================
# Save model
# ============================================================

def save_best_model(result):
    classifier_type = result["classifier"]

    filename = (
        f"best_{classifier_type}_"
        f"pca{result['components']}.joblib"
    )

    path = OUTPUT_DIR / filename

    model_data = {
        "classifier_type": classifier_type,
        "components": result["components"],
        "pca": result["pca"],
        "classifier": result["model"],
        "class_ids": result["class_ids"],
        "metrics": result["metrics"],
    }

    joblib.dump(
        model_data,
        path,
    )

    print(f"Saved {classifier_type} model to {path}")


# ============================================================
# Save results
# ============================================================

def save_results(all_results):
    rows = []

    for result in all_results:
        row = {
            "Classifier": result["classifier"],
            "PCA Components": result["components"],
            "Explained Variance": result["explained_variance"],
        }

        for metric, value in result["metrics"].items():
            row[metric.upper()] = value

        for metric, value in result["train_metrics"].items():
            row[f"Train {metric.upper()}"] = value

        rows.append(row)

    dataframe = pd.DataFrame(rows)

    dataframe.to_csv(
        OUTPUT_DIR / "classification_results.csv",
        index=False,
    )

    print("\nAll classification results:")
    print(dataframe.to_string(index=False))


def save_best_results(best_models):
    rows = []

    for classifier_type, result in best_models.items():
        row = {
            "Classifier": classifier_type,
            "PCA Components": result["components"],
            "Explained Variance": result["explained_variance"],
        }

        for metric, value in result["metrics"].items():
            row[metric.upper()] = value

        rows.append(row)

    dataframe = pd.DataFrame(rows)

    dataframe.to_csv(
        OUTPUT_DIR / "best_model_results.csv",
        index=False,
    )


# ============================================================
# PCA explained variance graph
# ============================================================

def plot_pca_variance(train_features):
    max_components = min(
        train_features.shape[0] - 1,
        train_features.shape[1],
    )

    pca = PCA(
        n_components=max_components,
        random_state=RANDOM_STATE,
    )

    pca.fit(train_features)

    cumulative_variance = np.cumsum(
        pca.explained_variance_ratio_
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        np.arange(1, len(cumulative_variance) + 1),
        cumulative_variance,
    )

    plt.xlabel("Number of PCA Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "pca_explained_variance.png",
        dpi=300,
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    set_seed()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    GRAPH_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Using device: {DEVICE}")

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    dataset = torch.utils.data.Subset  # Keep torchvision import usage explicit.

    image_dataset = __import__(
        "torchvision.datasets",
        fromlist=["ImageFolder"],
    ).ImageFolder(DATA_DIR)

    class_names = image_dataset.classes

    print(f"\nFound {len(image_dataset)} images.")
    print(f"Found {len(class_names)} classes:")
    print(class_names)

    # --------------------------------------------------------
    # Split dataset
    # --------------------------------------------------------

    train_indices, validation_indices = split_dataset(
        image_dataset
    )

    print(f"\nTraining images: {len(train_indices)}")
    print(f"Validation images: {len(validation_indices)}")
    print(
        f"Augmented training images: {len(train_indices) * 5}"
    )

    # --------------------------------------------------------
    # Extract or load ResNet18 features
    # --------------------------------------------------------

    (
        train_features,
        train_labels,
        validation_features,
        validation_labels,
    ) = load_or_precompute_features(
        image_dataset,
        train_indices,
        validation_indices,
    )

    print("\nFeature shapes:")
    print("Training:", train_features.shape)
    print("Validation:", validation_features.shape)

    plot_pca_variance(train_features)

    # --------------------------------------------------------
    # Train both classifiers across PCA configurations
    # --------------------------------------------------------

    all_results = []
    results_by_classifier = {
        classifier: []
        for classifier in CLASSIFIERS
    }

    for classifier_type in CLASSIFIERS:
        print(
            f"\n{'=' * 60}\n"
            f"Training {classifier_type.upper()}\n"
            f"{'=' * 60}"
        )

        for components in PCA_COMPONENTS:
            print(
                f"\nPCA components: {components}"
            )

            result = train_classifier(
                classifier_type=classifier_type,
                train_features=train_features,
                train_labels=train_labels,
                validation_features=validation_features,
                validation_labels=validation_labels,
                components=components,
            )

            all_results.append(result)

            results_by_classifier[
                classifier_type
            ].append(result)

            print(
                f"Accuracy:  {result['metrics']['accuracy']:.4f}"
            )

            print(
                f"CCE:       {result['metrics']['cce']:.4f}"
            )

            print(
                f"Precision: {result['metrics']['precision']:.4f}"
            )

            print(
                f"Recall:    {result['metrics']['recall']:.4f}"
            )

            print(
                f"F1:        {result['metrics']['f1']:.4f}"
            )

            print(
                f"FPR:       {result['metrics']['fpr']:.4f}"
            )

        # Generate PCA comparison graphs separately.
        plot_pca_comparison(
            results_by_classifier[classifier_type],
            classifier_type,
        )

        plot_loss_comparison(
            results_by_classifier[classifier_type],
            classifier_type,
        )

    # --------------------------------------------------------
    # Select best configuration for each classifier
    # --------------------------------------------------------

    best_models = {}

    for classifier_type, results in results_by_classifier.items():

        best_result = max(
            results,
            key=lambda result: (
                result["metrics"]["accuracy"],
                result["metrics"]["f1"],
                -result["metrics"]["cce"],
            ),
        )

        best_models[classifier_type] = best_result

        print(
            f"\nBest {classifier_type.title()} configuration:"
        )

        print(
            f"PCA Components: {best_result['components']}"
        )

        for metric, value in best_result["metrics"].items():
            print(
                f"{metric.upper()}: {value:.4f}"
            )

        save_best_model(best_result)

        # Only create these plots for the best configuration.
        plot_confusion_matrix(
            best_result,
            class_names,
        )

        plot_class_accuracy(
            best_result,
            class_names,
        )

    # --------------------------------------------------------
    # Generate final comparisons and reports
    # --------------------------------------------------------

    plot_best_model_comparison(best_models)

    save_results(all_results)

    save_best_results(best_models)

    print("\nFinished.")
    print(f"Results saved in: {OUTPUT_DIR}")
    print(f"Graphs saved in: {GRAPH_DIR}")


if __name__ == "__main__":
    main()