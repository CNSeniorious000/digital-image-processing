from __future__ import annotations

from typing import TYPE_CHECKING

from utils.frame import get_caller_name

if TYPE_CHECKING:
    from numpy import ndarray


def show(mat: ndarray):
    from cv2 import imshow, waitKey

    imshow(get_caller_name(), mat)
    waitKey(0)
