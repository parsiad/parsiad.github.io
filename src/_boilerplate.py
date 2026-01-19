from io import BytesIO
from base64 import b64encode

import matplotlib.pyplot as plt
from IPython.display import Markdown, display
from matplotlib.figure import Figure


def display_png_bytes(png_bytes: bytes):
    b64 = b64encode(png_bytes).decode()
    md = Markdown(f"![](data:image/png;base64,{b64})")
    display(md)


def display_fig(fig: Figure):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    display_png_bytes(buf.read())


def init():
    plt.style.use("seaborn-v0_8")
