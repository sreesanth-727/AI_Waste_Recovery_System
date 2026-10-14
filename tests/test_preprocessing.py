"""Unit tests for image preprocessing and input formatting."""

import numpy as np
from PIL import Image
from app import preprocess_image


def test_preprocess_image_resizing():
    """Verify that images of various arbitrary dimensions are resized to (1, 224, 224, 3)."""
    test_sizes = [(100, 100), (640, 480), (1024, 768), (224, 224), (50, 300)]

    for width, height in test_sizes:
        img = Image.new("RGB", (width, height), color=(120, 180, 200))
        processed = preprocess_image(img)

        assert isinstance(processed, np.ndarray)
        assert processed.shape == (1, 224, 224, 3)
        assert processed.dtype == np.float32


def test_preprocess_image_modes():
    """Verify that RGBA and Grayscale (L) modes are converted to 3-channel RGB."""
    # RGBA image
    rgba_img = Image.new("RGBA", (300, 300), color=(255, 0, 0, 128))
    processed_rgba = preprocess_image(rgba_img)
    assert processed_rgba.shape == (1, 224, 224, 3)

    # Grayscale image
    gray_img = Image.new("L", (200, 200), color=150)
    processed_gray = preprocess_image(gray_img)
    assert processed_gray.shape == (1, 224, 224, 3)
