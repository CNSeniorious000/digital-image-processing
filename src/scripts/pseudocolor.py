from numpy import uint8, zeros

from utils.read import read_grayscale
from utils.show import show


def pseudocolor(file: str):
    """
    Apply pseudocolor enhancement to a grayscale image based on specified RGB mapping functions.

    Args:
        file (str): Path to the input image
    """
    # Read grayscale image
    img = read_grayscale(file)

    # Create empty RGB channels
    height, width = img.shape
    pseudo_img = zeros((height, width, 3), dtype=uint8)

    # Convert img to float for calculations to avoid overflow
    img_float = img.astype(float)

    # Apply mapping functions for each channel
    # Red channel: Ramp function starting from middle
    red = (img_float >= 128) * (img_float - 128) * 2
    pseudo_img[:, :, 2] = red.clip(0, 255).astype(uint8)

    # Green channel: Triangle function
    green = (img_float <= 128) * img_float * 2 + (img_float > 128) * (510 - img_float * 2)
    pseudo_img[:, :, 1] = green.clip(0, 255).astype(uint8)

    # Blue channel: Inverted triangle function
    blue = (img_float <= 128) * img_float * 2 + (img_float > 128) * (510 - img_float * 2)
    pseudo_img[:, :, 0] = blue.clip(0, 255).astype(uint8)

    # Show original grayscale image
    show(img)
    # Show pseudocolor image
    show(pseudo_img)
