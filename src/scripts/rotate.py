import math
from math import cos, sin

from numpy import zeros_like

from utils.read import read_grayscale
from utils.show import show


def rotate(file: str, angle: float, radians=False):  # TODO: support specifying origin
    img = read_grayscale(file)
    print(img)

    if not radians:
        angle = angle * math.pi / 180  # convert to radians
        print(angle)

    target = zeros_like(img)

    width, height = img.shape

    cx, cy = width // 2, height // 2  # use center as origin b

    for y in range(height):
        for x in range(width):
            nx, ny = x - cx, y - cy
            dx = int(nx * cos(angle) + ny * sin(angle)) + cx
            dy = int(-nx * sin(angle) + ny * cos(angle)) + cy
            if 0 <= dx < width and 0 <= dy < height:
                target[y, x] = img[dy, dx]

    print(target)

    show(target)
