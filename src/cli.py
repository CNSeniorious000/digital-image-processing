from typer import Typer

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


if __name__ == "__main__":
    app()
