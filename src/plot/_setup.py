import logging
import shutil
from typing import Literal, TypedDict, Unpack

import matplotlib.pyplot as plt


class GlobalConfig(TypedDict, total=False):
    use_latex: bool
    font_family: Literal["serif", "sans-serif", "cursive", "fantasy", "monospace"]
    output_format: Literal["pdf", "png", "svg"]
    bbox: Literal["tight", "standard"]
    x_margin: float
    y_margin: float


def _setup(**kwargs: Unpack[GlobalConfig]):
    use_latex = kwargs.get("use_latex", True)
    if use_latex:
        if shutil.which("latex") is None:
            logging.warning("LaTeX is not available on this system. Falling back to non-LaTeX rendering.")
            use_latex = False
    plt.rcParams.update(
        {
            "text.usetex": use_latex,
            "text.latex.preamble": r"\usepackage{amsmath}",
            "font.family": kwargs.get("font_family", "serif"),
            "savefig.format": kwargs.get("output_format", "pdf"),
            "savefig.bbox": kwargs.get("bbox", "tight"),
            "axes.xmargin": kwargs.get("x_margin", 0.01),
            "axes.ymargin": kwargs.get("y_margin", 0.025),
        }
    )
