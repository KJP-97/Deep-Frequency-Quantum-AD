import cv2
import numpy as np
import random


def rotate_image(image, angle=None):
    """Rotate an MRI image."""
    if angle is None:
        angle = random.choice([-15, -10, 10, 15])

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        borderMode=cv2.BORDER_REFLECT
    )


def flip_image(image):
    """Random horizontal flipping."""
    return cv2.flip(image, 1)


def scale_image(image, scale_range=(0.9, 1.1)):
    """Random scaling."""
    scale = random.uniform(*scale_range)

    h, w = image.shape[:2]
    new_h = int(h * scale)
    new_w = int(w * scale)

    scaled = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    if scale >= 1:
        start_y = (new_h - h) // 2
        start_x = (new_w - w) // 2

        return scaled[
            start_y:start_y + h,
            start_x:start_x + w
        ]

    output = np.zeros_like(image)

    y = (h - new_h) // 2
    x = (w - new_w) // 2

    output[y:y + new_h, x:x + new_w] = scaled

    return output


def adjust_brightness_contrast(
    image,
    brightness_range=(-20, 20),
    contrast_range=(0.9, 1.1)
):
    """Random brightness and contrast adjustment."""

    brightness = random.uniform(*brightness_range)
    contrast = random.uniform(*contrast_range)

    result = image.astype(np.float32) * contrast + brightness

    return np.clip(result, 0, 255).astype(np.uint8)


def augment_image(image):
    """Apply random augmentation operations."""

    image = image.copy()

    if random.random() < 0.5:
        image = rotate_image(image)

    if random.random() < 0.5:
        image = flip_image(image)

    if random.random() < 0.5:
        image = scale_image(image)

    if random.random() < 0.5:
        image = adjust_brightness_contrast(image)

    return image
