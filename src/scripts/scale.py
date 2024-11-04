from enum import StrEnum

from numpy import zeros

from utils.read import read_grayscale
from utils.show import show


class Algorithm(StrEnum):
    nearest = "nearest"
    bilinear = "bilinear"


def scale(file: str, sx: float, sy: float, algorithm: Algorithm):
    img = read_grayscale(file)
    print(img)

    w, h = img.shape

    width, height = round(w * sx), round(h * sy)

    target = zeros((height, width), dtype=img.dtype)

    for y in range(height):
        for x in range(width):
            src_x = round(x / sx)
            src_y = round(y / sy)
            if 0 <= src_x < w and 0 <= src_y < h:
                if algorithm == Algorithm.nearest:
                    target[y, x] = img[src_y, src_x]
                elif algorithm == Algorithm.bilinear:
                    src_x1 = min(src_x + 1, w - 1)
                    src_y1 = min(src_y + 1, h - 1)

                    dx = x / sx - src_x
                    dy = y / sy - src_y

                    target[y, x] = (
                        img[src_y, src_x] * (1 - dx) * (1 - dy)
                        + img[src_y, src_x1] * dx * (1 - dy)
                        + img[src_y1, src_x] * (1 - dx) * dy
                        + img[src_y1, src_x1] * dx * dy
                    )

    print(target)

    show(target)
