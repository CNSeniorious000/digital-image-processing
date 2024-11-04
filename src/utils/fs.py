from pathlib import Path

root = Path(__file__, "../../../data").resolve()


def resolve(path: str):
    if Path(path).exists():
        return path

    elif (p := Path(root / path)).exists():
        return str(p)

    raise FileNotFoundError(path)
