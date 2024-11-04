from numpy import zeros_like

from utils.read import read_grayscale
from utils.show import show


def translate(file: str, dx: int, dy: int):  # TODO: support float dx, dy
    img = read_grayscale(file)
    print(img)

    target = zeros_like(img)

    width, height = img.shape

    for y in range(height):
        for x in range(width):
            if 0 <= x + dx < width and 0 <= y + dy < height:
                target[y + dy, x + dx] = img[y, x]

    print(target)

    show(target)
