from typing import Literal

import cv2
import numpy as np

from utils.read import read_grayscale
from utils.show import show


def get_spectrum(image: np.ndarray):
    # Apply DFT
    dft = cv2.dft(image.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Calculate magnitude spectrum
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    spectrum = 20 * np.log(magnitude + 1)
    print(spectrum)

    # Normalize for display
    spectrum = cv2.normalize(spectrum, spectrum, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return dft_shift, spectrum


def create_ideal_filter(shape, d0: int, filter_type: Literal["low", "high"] = "low"):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    # Create meshgrid for distance calculation
    u = np.arange(rows) - crow
    v = np.arange(cols) - ccol
    u_grid, v_grid = np.meshgrid(v, u)
    dist = np.sqrt(u_grid**2 + v_grid**2)

    # Create filter
    match filter_type:
        case "low":
            mask = (dist <= d0).astype(np.float32)
        case "high":
            mask = (dist > d0).astype(np.float32)
        case _:
            raise ValueError(filter_type)

    return np.stack([mask, mask], axis=2)


def create_butterworth_filter(shape, d0: int, filter_type: Literal["low", "high"] = "low"):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    u = np.arange(rows) - crow
    v = np.arange(cols) - ccol
    u_grid, v_grid = np.meshgrid(v, u)
    dist = np.sqrt(u_grid**2 + v_grid**2)

    match filter_type:
        case "low":
            mask = 1 / (1 + (dist / d0) ** 2)
        case "high":
            mask = 1 / (1 + (d0 / dist) ** 2)
            mask[crow, ccol] = 0  # Fix division by zero
        case _:
            raise ValueError(filter_type)

    return np.stack([mask, mask], axis=2)


def apply_frequency_filter(image: np.ndarray, filter_mask: np.ndarray):
    dft_shift, _ = get_spectrum(image)

    # Apply filter
    filtered_dft = dft_shift * filter_mask

    # Inverse DFT
    idft_shift = np.fft.ifftshift(filtered_dft)
    img_back = cv2.idft(idft_shift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Normalize
    img_back = cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return img_back


def freq_domain(file: str, d0: int = 50):
    """Apply frequency domain filters to image"""
    img = read_grayscale(file)

    # Get original spectrum
    _, spectrum = get_spectrum(img)
    show(spectrum)

    # Create and display filters
    ideal_low = create_ideal_filter(img.shape, d0, "low")
    ideal_high = create_ideal_filter(img.shape, d0, "high")
    butter_low = create_butterworth_filter(img.shape, d0, "low")
    butter_high = create_butterworth_filter(img.shape, d0, "high")

    # Display filters
    show((ideal_low[:, :, 0] * 255).astype(np.uint8))
    show((ideal_high[:, :, 0] * 255).astype(np.uint8))
    show((butter_low[:, :, 0] * 255).astype(np.uint8))
    show((butter_high[:, :, 0] * 255).astype(np.uint8))

    # Apply filters and show results
    ideal_low_result = apply_frequency_filter(img, ideal_low)
    ideal_high_result = apply_frequency_filter(img, ideal_high)
    butter_low_result = apply_frequency_filter(img, butter_low)
    butter_high_result = apply_frequency_filter(img, butter_high)

    show(ideal_low_result)
    show(ideal_high_result)
    show(butter_low_result)
    show(butter_high_result)
