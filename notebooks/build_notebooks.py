"""Turn the lesson sources (notebooks/lessons/*.py, "percent" format) into Jupyter notebooks.

  python notebooks/build_notebooks.py                         # build all lessons into notebooks/*.ipynb
  python notebooks/build_notebooks.py --execute 02_sound_to_spectrogram   # build, then run it and keep outputs

Percent format: "# %%" starts a code cell, "# %% [markdown]" starts a text cell whose lines begin with "# ".
"""
import argparse
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
LESSONS = HERE / "lessons"


def parse_percent(text: str) -> list[tuple[str, str]]:
    cells, kind, lines = [], None, []

    def flush():
        if kind is not None:
            body = "\n".join(lines).strip("\n")
            if body.strip():
                cells.append((kind, body))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, lines = ("markdown" if "[markdown]" in line else "code"), []
        elif kind == "markdown":
            lines.append(line[2:] if line.startswith("# ") else line.lstrip("#"))
        elif kind is not None:
            lines.append(line)
    flush()
    return cells


def build(lesson: Path, out_dir: Path) -> Path:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for kind, src in parse_percent(lesson.read_text(encoding="utf-8")):
        nb.cells.append(nbformat.v4.new_markdown_cell(src) if kind == "markdown" else nbformat.v4.new_code_cell(src))
    out = Path(out_dir) / f"{lesson.stem}.ipynb"
    nbformat.write(nb, out)
    return out


def build_all(lessons_dir: Path = LESSONS, out_dir: Path = HERE) -> list[Path]:
    return [build(p, out_dir) for p in sorted(Path(lessons_dir).glob("[0-9][0-9]_*.py"))]


def execute(name: str, timeout: int = 900) -> Path:
    from nbclient import NotebookClient
    path = HERE / f"{name}.ipynb"
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
    nbformat.write(nb, path)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--execute", nargs="*", default=[], help="lesson names to run after building")
    a = p.parse_args()
    for path in build_all():
        print("built", path.name)
    for name in a.execute:
        print("executing", name, "...")
        execute(name)
        print("  ok")


if __name__ == "__main__":
    main()
