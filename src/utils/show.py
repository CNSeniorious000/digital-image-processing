from __future__ import annotations

from typing import TYPE_CHECKING

from utils.frame import get_last_frame_name

if TYPE_CHECKING:
    from numpy import ndarray


def show(mat: ndarray):
    from cv2 import imshow, waitKey

    imshow(get_last_frame_name(), mat)
    waitKey(0)
