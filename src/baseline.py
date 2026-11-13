"""
Evaluate the baseline performance for a classifier that just guesses the most common class for every input.
"""
from collections import Counter
from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "pc_parts"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".pgm", ".png", ".ppm", ".tif", ".tiff", ".webp"}

def compute_avg_metrics(num_classes, most_common_count, total_image_count):
    # compute metrics for the baseline predicted class
    precision = most_common_count / total_image_count # all items predicted positive
    recall = 1 # there are no false negatives
    fpr = 1 # there are no true negatives
    f1 = 2 * precision * recall / (precision + recall)

    # the metrics are all zero for the ten other classes

    # "average" between all the classes
    return precision / num_classes, recall / num_classes, fpr / num_classes, f1 / num_classes

def compute_aggregate(most_common_count, total_image_count):
    acc = most_common_count / total_image_count
    err = 1 - acc
    cce = -np.log(acc)
    return acc, err, cce

def main():
    class_counts = Counter({
        class_dir.name: sum(
            path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            for path in class_dir.rglob("*")
        )
        for class_dir in DATA_DIR.iterdir()
        if class_dir.is_dir()
    })
    if not class_counts:
        raise ValueError(f"No class directories found in {DATA_DIR}")

    num_classes = len(class_counts)
    most_common_class, most_common_count = class_counts.most_common(1)[0]
    total_image_count = sum(class_counts.values())

    print(f"Total dataset images: {total_image_count}")
    print(f"Most common class: {most_common_class} ({most_common_count} images)")

    # compute metrics for each class index
    precision, recall, fpr, f1 = compute_avg_metrics(num_classes, most_common_count, total_image_count)
    print (f"\nAverage precision: {precision:.4f}")
    print (f"Average recall: {recall:.4f}")
    print (f"Average false positive rate: {fpr:.4f}")
    print (f"Average F1: {f1:.4f}")

    # aggregate metrics
    acc, err, cce = compute_aggregate(most_common_count, total_image_count)
    print (f"\nAccuracy: {acc * 100:.2f}%")
    print (f"Error Rate: {err * 100:.2f}%")
    print (f"Categorical Cross-Entropy: {cce:.4f}")


if __name__ == "__main__":
    main()