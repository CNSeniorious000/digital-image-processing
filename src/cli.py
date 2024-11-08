from typer import Typer

from scripts.scale import Algorithm
from utils.log import patch_print

patch_print()

app = Typer(
    no_args_is_help=True,
    help="Image Processing CLI Entry",
    add_completion=False,
)


@app.command()
def translate(dx: int, dy: int, file: str = "lena.bmp"):
    """Translate an image by dx and dy"""

    print(f"Translating {file} by ({dx}, {dy})")

    from scripts.translate import translate

    translate(file, dx, dy)


@app.command()
def rotate(angle: float, file: str = "lena.bmp"):
    """Rotate an image by angle"""

    print(f"Rotating {file} by {angle} degrees")

    from scripts.rotate import rotate

    rotate(file, angle)


@app.command()
def scale(
    sx: float,
    sy: float,
    algorithm: Algorithm = Algorithm.bilinear,
    file: str = "lena.bmp",
):
    """Scale an image by sx and sy"""

    print(f"Scaling {file} by ({sx}, {sy})")

    from scripts.scale import scale

    scale(file, sx, sy, algorithm)


if __name__ == "__main__":
    app()
