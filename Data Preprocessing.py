import cv2
import numpy as np
from pathlib import Path


def resize_image(image, size=(224, 224)):
    """Resize MRI image to the required input size."""
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def gaussian_filter(image, kernel_size=(5, 5)):
    """Reduce image noise using Gaussian filtering."""
    return cv2.GaussianBlur(image, kernel_size, 0)


def normalize_image(image):
    """Normalize pixel intensity to [0, 1]."""
    image = image.astype(np.float32)
    min_val = np.min(image)
    max_val = np.max(image)

    if max_val - min_val == 0:
        return np.zeros_like(image)

    return (image - min_val) / (max_val - min_val)


def apply_clahe(image):
    """Enhance local contrast using CLAHE."""
    image_uint8 = np.uint8(image * 255)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(image_uint8)

    return enhanced.astype(np.float32) / 255.0


def skull_stripping(image):
    """
    Basic foreground extraction for MRI preprocessing.
    This function can be replaced by the exact skull-stripping
    algorithm used in the experimental setup.
    """
    image_uint8 = np.uint8(image * 255)

    _, mask = cv2.threshold(
        image_uint8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8)
    )

    stripped = image * (mask / 255.0)

    return stripped


def preprocess_mri(image_path, output_size=(224, 224)):
    """Complete MRI preprocessing pipeline."""

    # Load image
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    # 1. Resize
    image = resize_image(image, output_size)

    # 2. Normalize
    image = normalize_image(image)

    # 3. Gaussian filtering
    image = gaussian_filter(image)

    # 4. CLAHE
    image = apply_clahe(image)

    # 5. Skull stripping / background removal
    image = skull_stripping(image)

    # Final normalization
    image = normalize_image(image)

    return image


if __name__ == "__main__":

    image_path = "data/sample_mri.jpg"

    processed_image = preprocess_mri(image_path)

    output_path = "data/preprocessed_mri.png"

    cv2.imwrite(
        output_path,
        np.uint8(processed_image * 255)
    )

    print("Preprocessing completed.")
    print(f"Saved to: {output_path}")
