from .fs import resolve


def read_grayscale(path: str):
    from cv2 import IMREAD_GRAYSCALE, imread

    return imread(resolve(path), flags=IMREAD_GRAYSCALE)
