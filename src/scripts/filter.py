from enum import StrEnum

from numpy import zeros_like

from utils.read import read_grayscale
from utils.show import show


class Padding(StrEnum):
    zero = "zero"
    copy = "copy"
    mirror = "mirror"


def mean_filter(file: str, size: int, padding: Padding):
    img = read_grayscale(file)
    img = img / 255
    print(img)

    target = zeros_like(img)

    width, height = img.shape

    count = size**2
    r = size // 2

    for y in range(height):
        for x in range(width):
            total = 0

            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    px = x + dx
                    py = y + dy

                    if padding == Padding.zero:
                        if not (0 <= px < width and 0 <= py < height):
                            continue
                    elif padding == Padding.copy:
                        px = max(0, min(px, width - 1))
                        py = max(0, min(py, height - 1))
                    elif padding == Padding.mirror:
                        if px < 0:
                            px = -px
                        elif px >= width:
                            px = 2 * width - px - 1
                        if py < 0:
                            py = -py
                        elif py >= height:
                            py = 2 * height - py - 1

                    total += img[py, px]

            target[y, x] = total / count

    print(target)

    show(target)
