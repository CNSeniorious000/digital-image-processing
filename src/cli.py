from typer import Argument, Typer

from scripts.filter import Padding
from scripts.freq import freq_domain
from scripts.scale import Algorithm
from utils.log import patch_print

patch_print()

app = Typer(
    no_args_is_help=True,
    help="Image Processing CLI Entry",
    add_completion=False,
)

LENA = "lena.bmp"


@app.command()
def translate(dx: int, dy: int, file: str = LENA):
    """Translate an image by dx and dy"""

    print(f"Translating {file} by ({dx}, {dy})")

    from scripts.translate import translate

    translate(file, dx, dy)


@app.command()
def rotate(angle: float, file: str = LENA):
    """Rotate an image by angle"""

    print(f"Rotating {file} by {angle} degrees")

    from scripts.rotate import rotate

    rotate(file, angle)


@app.command()
def scale(
    sx: float,
    sy: float,
    algorithm: Algorithm = Algorithm.bilinear,
    file: str = LENA,
):
    """Scale an image by sx and sy"""

    print(f"Scaling {file} by ({sx}, {sy})")

    from scripts.scale import scale

    scale(file, sx, sy, algorithm)


@app.command()
def mean(
    size: int,
    padding: Padding = Padding.zero,
    file: str = LENA,
):
    """Apply a mean filter to an image"""

    print(f"Applying mean filter to {file} with size {size} and padding {padding}")

    from scripts.filter import mean_filter

    mean_filter(file, size, padding)


@app.command()
def median(
    size: int,
    padding: Padding = Padding.zero,
    file: str = LENA,
):
    """Apply a median filter to an image"""

    print(f"Applying median filter to {file} with size {size} and padding {padding}")

    from scripts.filter import median_filter

    median_filter(file, size, padding)


@app.command()
def prewitt(file: str = LENA):
    """Apply Prewitt operators to an image"""

    print(f"Applying Prewitt operators to {file}")

    from scripts.conv import prewitt

    prewitt(file)


@app.command()
def sobel(file: str = LENA):
    """Apply Sobel operators to an image"""

    print(f"Applying Sobel operators to {file}")

    from scripts.conv import sobel

    sobel(file)


@app.command()
def laplacian(file: str = LENA, neighbors: int = Argument(8, help="4 or 8")):
    """Apply Laplacian operator to an image"""

    print(f"Applying Laplacian operator to {file}")

    from scripts.conv import laplacian

    laplacian(file, neighbors)  # type: ignore


@app.command()
def sharpen(file: str = LENA, neighbors: int = Argument(8, help="4 or 8")):
    """Apply Laplacian operator to sharpen an image"""

    print(f"Applying Laplacian operator to sharpen {file}")

    from scripts.conv import laplacian

    laplacian(file, neighbors, sharpen=True)  # type: ignore


@app.command()
def freq(file: str = LENA, d0: int = Argument(50, help="Cutoff frequency")):
    """Apply frequency domain filters to an image"""
    freq_domain(file, d0)


@app.command()
def pseudocolor(file: str = LENA):
    """Apply pseudocolor enhancement to a grayscale image"""

    print(f"Applying pseudocolor enhancement to {file}")

    from scripts.pseudocolor import pseudocolor

    pseudocolor(file)


if __name__ == "__main__":
    app()
