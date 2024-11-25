from typing import Literal

import numpy as np

from utils.read import read_grayscale
from utils.show import show


def apply_convolution(image: np.ndarray, kernel: np.ndarray):
    assert kernel.shape == (3, 3), kernel.shape

    # out = np.zeros_like(image)
    # w, h = image.shape
    # for i in range(1, w - 1):
    #     for j in range(1, h - 1):
    #         out[i, j] = np.sum(image[i - 1 : i + 2, j - 1 : j + 2] * kernel)

    # numpy way of doing the same thing
    window = np.lib.stride_tricks.sliding_window_view(image, (3, 3))
    out = np.pad(np.einsum("ijkl,kl->ij", window, kernel), 1, mode="constant")

    return out


# Prewitt operators
prewitt_x = np.array(
    [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
    ]
)
prewitt_y = np.array(
    [
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1],
    ]
)


def prewitt(file: str):
    img = read_grayscale(file) / 255
    print(img)

    prewitt_grad_x = apply_convolution(img, prewitt_x)
    prewitt_grad_y = apply_convolution(img, prewitt_y)
    prewitt_gradient = np.sqrt(prewitt_grad_x**2 + prewitt_grad_y**2)

    print(prewitt_gradient)

    show(prewitt_gradient)


# Sobel operators
sobel_x = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ]
)
sobel_y = np.array(
    [
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]
)


def sobel(file: str):
    img = read_grayscale(file) / 255
    print(img)

    sobel_grad_x = apply_convolution(img, sobel_x)
    sobel_grad_y = apply_convolution(img, sobel_y)
    sobel_gradient = np.sqrt(sobel_grad_x**2 + sobel_grad_y**2)

    print(sobel_gradient)

    show(sobel_gradient)


# Laplacian operators
laplacian_4 = -np.array(
    [
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ]
)
laplacian_8 = -np.array(
    [
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1],
    ]
)


def laplacian(file: str, neighbors: Literal[4, 8], sharpen=False):
    img = read_grayscale(file) / 255
    print(img)

    kernel = laplacian_4 if neighbors == 4 else laplacian_8

    laplacian_gradient = apply_convolution(img, kernel)

    target = (img + laplacian_gradient) if sharpen else laplacian_gradient

    print(target)

    show(target)
