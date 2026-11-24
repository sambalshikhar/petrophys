"""The object every plot returns: its figure and axes, shown and saved the same way."""

from __future__ import annotations

from typing import Any, List, Optional


class Plot:
    """A drawn plot.

    ``fig`` is the Matplotlib figure, ``ax`` its main axes and ``axes`` all of
    them (panels, tracks), to style further with Matplotlib. At the end of a
    notebook cell the plot shows its figure; ``.save(path)`` writes it as PNG,
    PDF, SVG or any format Matplotlib knows by the extension, and ``.show()``
    displays it from a script.

    Examples
    --------
    >>> chart = pp.bar_chart(["A", "B"], [3, 5])            # doctest: +SKIP
    >>> chart.ax.set_ylim(0, 10); chart.save("pay.png")     # doctest: +SKIP
    """

    def __init__(self, fig: Any = None, ax: Any = None):
        self.fig = fig
        panels: Optional[List[Any]] = None
        if ax is not None and not hasattr(ax, "plot"):  # several axes
            import numpy as np

            panels = [item for item in np.ravel(np.asarray(ax, dtype=object)) if item is not None]
            ax = panels[0] if panels else None
        if ax is None and fig is not None and fig.axes:
            ax = fig.axes[0]
        self.ax = ax
        self._panels = panels

    @property
    def axes(self) -> List[Any]:
        """Every axes of the plot: the panels or tracks given, else the figure's."""
        if self._panels is not None:
            return list(self._panels)
        return list(self.fig.axes) if self.fig is not None else []

    def show(self) -> None:
        """Display the figure (in a script, a window; in a notebook, inline)."""
        import matplotlib.pyplot as plt

        if self.fig is None:
            raise RuntimeError("The plot has no figure")
        plt.show()

    def save(self, path, dpi: int = 150) -> None:
        """Save the figure, its format from the extension of ``path``."""
        from petrophys.visualization.style import save_figure

        if self.fig is None:
            raise RuntimeError("The plot has no figure")
        save_figure(self.fig, path, dpi=dpi)

    def close(self) -> None:
        """Release the figure (frees memory in loops)."""
        import matplotlib.pyplot as plt

        if self.fig is not None:
            plt.close(self.fig)

    def _ipython_display_(self) -> None:
        # A plot at the end of a notebook cell shows its figure, once.
        import matplotlib.pyplot as plt
        from IPython.core.pylabtools import print_figure
        from IPython.display import Image, display

        if self.fig is None:
            return
        display(Image(data=print_figure(self.fig, "png", bbox_inches="tight")))
        plt.close(self.fig)  # keeps .fig for saving; the notebook does not draw it again

    def __repr__(self) -> str:
        title = ""
        if self.ax is not None:
            title = self.ax.get_title() or (self.fig._suptitle.get_text() if getattr(self.fig, "_suptitle", None) else "")
        return f"{type(self).__name__}({title!r}, {len(self.axes)} axes)" if title else f"{type(self).__name__}()"


__all__ = ["Plot"]
