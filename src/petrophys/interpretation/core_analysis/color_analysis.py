"""
Core image color analysis from optical scanning.

Loads RGB and HSV color measurements extracted from continuous core
slab photographs and computes brightness/color indices used as
lithology proxies in digital core workflows.

References
----------
Smith, A.R. (1978). Color gamut transform pairs. *Computer Graphics*
(SIGGRAPH '78) 12(3), 12-19.
"""

from typing import Optional, Union, Dict, Any
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np

from petrophys._internal.statistics import finite_min_max

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import lasio
    HAS_LASIO = True
except ImportError:
    HAS_LASIO = False


# Simple color-to-lithology heuristics based on RGB/HSV
# Hue ranges (0–360°): red~0, yellow~60, green~120, cyan~180, blue~240, magenta~300
_COLOR_CLASSES = {
    "dark_organic":   {"brightness_max": 0.25},
    "red_iron":       {"hue_min": 340, "hue_max": 20,  "saturation_min": 0.3},
    "tan_sand":       {"hue_min": 20,  "hue_max": 50,  "brightness_min": 0.5},
    "grey_carbonate": {"saturation_max": 0.15, "brightness_min": 0.4},
    "green_chlorite": {"hue_min": 80,  "hue_max": 160, "saturation_min": 0.2},
}


@dataclass
class CoreImageData:
    """
    Color measurements extracted from core slab photographs.

    Parameters
    ----------
    depth : np.ndarray
        Measured depth (m or ft).
    grey_level : np.ndarray, optional
        Average grey level (0–255 or 0–1).
    grey_std : np.ndarray, optional
        Standard deviation of grey level (texture proxy).
    r : np.ndarray, optional
        Red channel (0–255).
    g : np.ndarray, optional
        Green channel (0–255).
    b : np.ndarray, optional
        Blue channel (0–255).
    hue : np.ndarray, optional
        Hue (0–360°).
    saturation : np.ndarray, optional
        Saturation (0–1).
    value : np.ndarray, optional
        Value/brightness in HSV (0–1).
    metadata : dict

    Examples
    --------
    >>> img = CoreImageData.from_las("core_image.las")
    >>> bi = img.brightness_index()
    >>> ci = img.color_index()
    """

    depth: np.ndarray
    grey_level: Optional[np.ndarray] = None
    grey_std: Optional[np.ndarray] = None
    r: Optional[np.ndarray] = None
    g: Optional[np.ndarray] = None
    b: Optional[np.ndarray] = None
    hue: Optional[np.ndarray] = None
    saturation: Optional[np.ndarray] = None
    value: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---- Properties ----

    @property
    def n_samples(self) -> int:
        return len(self.depth)

    @property
    def top(self) -> Optional[float]:
        return finite_min_max(self.depth)[0]

    @property
    def bottom(self) -> Optional[float]:
        return finite_min_max(self.depth)[1]

    @property
    def has_rgb(self) -> bool:
        return self.r is not None and self.g is not None and self.b is not None

    @property
    def has_hsv(self) -> bool:
        return (self.hue is not None
                and self.saturation is not None
                and self.value is not None)

    # ---- Loaders ----

    @classmethod
    def from_las(
        cls,
        filepath: Union[str, Path],
        grey_mnemonic: Optional[str] = "E_MBWP",
        grey_std_mnemonic: Optional[str] = "E_STDP",
        r_mnemonic: Optional[str] = "E_MRPX",
        g_mnemonic: Optional[str] = "E_MGPX",
        b_mnemonic: Optional[str] = "E_MBPX",
        hue_mnemonic: Optional[str] = "E_MHPX",
        sat_mnemonic: Optional[str] = "E_MSPX",
        val_mnemonic: Optional[str] = "E_MVPX",
        null_value: float = -999.25,
    ) -> "CoreImageData":
        """
        Load color data from a LAS file.

        Defaults match the EpsLog/Stratum Reservoir ``Image_10.0mm_WL.las``
        mnemonic convention (``E_MBWP``, ``E_MRPX``, etc.).

        EpsLog stores RGB, grey level, saturation, and value on a **0–255**
        scale and hue in **degrees (0–360)**. This loader normalises
        saturation and value to 0–1 automatically when their stored maximum
        exceeds 1.5, so that ``color_index()`` and ``lithology_from_color()``
        work correctly regardless of source scale.

        Depth is read via ``lasio``'s ``las.index`` property (always the
        first column, regardless of mnemonic name).

        Parameters
        ----------
        filepath : str or Path
        grey_mnemonic : str, optional
            Average grey level mnemonic (EpsLog: ``E_MBWP``).
        grey_std_mnemonic : str, optional
            Grey level standard deviation (EpsLog: ``E_STDP``).
        r_mnemonic, g_mnemonic, b_mnemonic : str, optional
            RGB channel mnemonics (EpsLog: ``E_MRPX``, ``E_MGPX``, ``E_MBPX``).
        hue_mnemonic, sat_mnemonic, val_mnemonic : str, optional
            HSV channel mnemonics (EpsLog: ``E_MHPX``, ``E_MSPX``, ``E_MVPX``).
            Hue is kept in degrees; saturation and value are normalised to 0–1.
        null_value : float
        """
        if not HAS_LASIO:
            raise ImportError("lasio is required: pip install lasio")

        las = lasio.read(str(filepath))

        def _get(mn):
            if mn is None or mn not in las.keys():
                return None
            arr = np.array(las[mn], dtype=float)
            arr[arr == null_value] = np.nan
            return arr

        def _norm01(arr):
            """Scale 0-255 values to [0,1] if the stored range clearly exceeds 1."""
            if arr is None:
                return None
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return arr
            mx = np.nanmax(finite)
            return arr / 255.0 if mx > 1.5 else arr

        depth = np.array(las.index, dtype=float)
        meta = {
            "well": las.well.WELL.value if hasattr(las.well, "WELL") else "",
            "source": str(filepath),
        }

        # Saturation and value: normalise to [0,1] (EpsLog stores 0-255).
        # Hue stays in degrees. Grey and RGB stay in raw 0-255 (used as-is
        # in rgb_to_hsv and normalised inside brightness_index).
        return cls(
            depth=depth,
            grey_level=_get(grey_mnemonic),
            grey_std=_get(grey_std_mnemonic),
            r=_get(r_mnemonic),
            g=_get(g_mnemonic),
            b=_get(b_mnemonic),
            hue=_get(hue_mnemonic),
            saturation=_norm01(_get(sat_mnemonic)),
            value=_norm01(_get(val_mnemonic)),
            metadata=meta,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: "pd.DataFrame",
        depth_col: str = "DEPTH",
        grey_col: Optional[str] = None,
        grey_std_col: Optional[str] = None,
        r_col: Optional[str] = None,
        g_col: Optional[str] = None,
        b_col: Optional[str] = None,
        hue_col: Optional[str] = None,
        sat_col: Optional[str] = None,
        val_col: Optional[str] = None,
    ) -> "CoreImageData":
        """Create CoreImageData from a pandas DataFrame."""
        if not HAS_PANDAS:
            raise ImportError("pandas is required")

        def _col(name):
            if name and name in df.columns:
                return df[name].values.astype(float)
            return None

        return cls(
            depth=df[depth_col].values.astype(float),
            grey_level=_col(grey_col),
            grey_std=_col(grey_std_col),
            r=_col(r_col),
            g=_col(g_col),
            b=_col(b_col),
            hue=_col(hue_col),
            saturation=_col(sat_col),
            value=_col(val_col),
        )

    # ---- Indices ----

    def brightness_index(self) -> np.ndarray:
        """
        Brightness index scaled to [0, 1].

        Uses grey_level if available, otherwise derives from RGB mean,
        or falls back to the HSV value channel. Grey level and RGB are
        divided by their maximum in the data, so 1 is the brightest sample
        loaded, not absolute white.

        Returns
        -------
        np.ndarray
            Low = dark (organic-rich), high = light (clean carbonate/evaporite).
        """
        if self.grey_level is not None:
            raw = self.grey_level
            finite = raw[np.isfinite(raw)]
            if finite.size == 0:
                return raw.astype(float, copy=True)
            mx = np.nanmax(finite)
            return raw / mx if mx > 0 else raw

        if self.has_rgb:
            raw = (self.r + self.g + self.b) / 3.0
            finite = raw[np.isfinite(raw)]
            if finite.size == 0:
                return raw.astype(float, copy=True)
            mx = np.nanmax(finite)
            return raw / mx if mx > 0 else raw

        if self.value is not None:
            return self.value

        return np.full(self.n_samples, np.nan)

    def color_index(self) -> np.ndarray:
        """
        Color index: the HSV saturation, or (max - min) / max of RGB.

        Highlights color-rich intervals (iron staining, chlorite, etc.)
        against grey/white intervals. Returns 0 for grey, 1 for maximum color.

        Returns
        -------
        np.ndarray
            Higher values = more colorful / chromatic.
        """
        if self.has_hsv:
            # Saturation directly reflects color intensity
            return np.where(np.isnan(self.saturation), np.nan, self.saturation)

        if self.has_rgb:
            # Approximate saturation from RGB: (max - min) / max
            stack = np.stack([self.r, self.g, self.b], axis=1)
            finite_rows = np.isfinite(stack).any(axis=1)
            rgb_max = np.full(stack.shape[0], np.nan, dtype=float)
            rgb_min = np.full(stack.shape[0], np.nan, dtype=float)
            if np.any(finite_rows):
                rgb_max[finite_rows] = np.nanmax(stack[finite_rows], axis=1)
                rgb_min[finite_rows] = np.nanmin(stack[finite_rows], axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(rgb_max > 0, (rgb_max - rgb_min) / rgb_max, np.nan)

        return np.full(self.n_samples, np.nan)

    def texture_index(self) -> np.ndarray:
        """
        Texture index from grey level standard deviation.

        Higher values indicate heterogeneous textures (lamination,
        bioturbation, vugs). Returns NaN if grey_std is not available.
        """
        if self.grey_std is None:
            return np.full(self.n_samples, np.nan)
        finite = self.grey_std[np.isfinite(self.grey_std)]
        if finite.size == 0:
            return self.grey_std.astype(float, copy=True)
        mx = np.nanmax(finite)
        return self.grey_std / mx if mx > 0 else self.grey_std

    def rgb_to_hsv(self) -> Dict[str, np.ndarray]:
        """
        Convert RGB (0–255) to HSV.

        Returns
        -------
        dict with keys "hue" (0–360), "saturation" (0–1), "value" (0–1).
        Raises ValueError if RGB channels are not loaded.
        """
        if not self.has_rgb:
            raise ValueError("RGB channels (r, g, b) are required")

        r = self.r / 255.0
        g = self.g / 255.0
        b = self.b / 255.0

        cmax = np.maximum.reduce([r, g, b])
        cmin = np.minimum.reduce([r, g, b])
        delta = cmax - cmin

        # Value
        v = cmax

        # Saturation
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(cmax > 0, delta / cmax, 0.0)

        # Hue
        h = np.zeros_like(r)
        mask_r = (cmax == r) & (delta > 0)
        mask_g = (cmax == g) & (delta > 0)
        mask_b = (cmax == b) & (delta > 0)

        with np.errstate(divide="ignore", invalid="ignore"):
            h = np.where(mask_r, ((g - b) / delta) % 6, h)
            h = np.where(mask_g, (b - r) / delta + 2, h)
            h = np.where(mask_b, (r - g) / delta + 4, h)

        h = h * 60.0  # degrees

        # Missing RGB samples have no hue or saturation
        missing = ~(np.isfinite(r) & np.isfinite(g) & np.isfinite(b))
        h = np.where(missing, np.nan, h)
        s = np.where(missing, np.nan, s)

        return {"hue": h, "saturation": s, "value": v}

    def lithology_from_color(self) -> np.ndarray:
        """
        Simple rule-based lithology classification from color.

        Returns
        -------
        np.ndarray of str
            One of: "dark_organic", "red_iron", "tan_sand",
            "grey_carbonate", "green_chlorite", "unclassified".
        """
        bi = self.brightness_index()
        result = np.full(self.n_samples, "unclassified", dtype=object)

        # Dark / organic-rich
        result[bi < 0.25] = "dark_organic"

        if self.has_hsv:
            h = self.hue
            s = self.saturation

            # Grey carbonate: low saturation, not dark
            grey_mask = (s < 0.15) & (bi >= 0.4)
            result[grey_mask] = "grey_carbonate"

            # Red/iron staining: hue near red
            red_mask = ((h <= 20) | (h >= 340)) & (s >= 0.3) & (bi >= 0.25)
            result[red_mask] = "red_iron"

            # Tan sand: warm yellow-orange hue
            tan_mask = (h > 20) & (h <= 50) & (bi >= 0.5)
            result[tan_mask] = "tan_sand"

            # Green chlorite
            green_mask = (h > 80) & (h <= 160) & (s >= 0.2)
            result[green_mask] = "green_chlorite"

        elif self.has_rgb:
            hsv = self.rgb_to_hsv()
            h, s = hsv["hue"], hsv["saturation"]

            grey_mask = (s < 0.15) & (bi >= 0.4)
            result[grey_mask] = "grey_carbonate"

            red_mask = ((h <= 20) | (h >= 340)) & (s >= 0.3) & (bi >= 0.25)
            result[red_mask] = "red_iron"

            tan_mask = (h > 20) & (h <= 50) & (bi >= 0.5)
            result[tan_mask] = "tan_sand"

            green_mask = (h > 80) & (h <= 160) & (s >= 0.2)
            result[green_mask] = "green_chlorite"

        return result

    # ---- Utilities ----

    def filter_by_depth(self, top: float, bottom: float) -> "CoreImageData":
        """Return a new CoreImageData filtered to [top, bottom]."""
        mask = (self.depth >= top) & (self.depth <= bottom)

        def _mask(arr):
            return arr[mask] if arr is not None else None

        return CoreImageData(
            depth=self.depth[mask],
            grey_level=_mask(self.grey_level),
            grey_std=_mask(self.grey_std),
            r=_mask(self.r),
            g=_mask(self.g),
            b=_mask(self.b),
            hue=_mask(self.hue),
            saturation=_mask(self.saturation),
            value=_mask(self.value),
            metadata=dict(self.metadata),
        )

    def to_dataframe(self) -> "pd.DataFrame":
        """Export all loaded channels to a pandas DataFrame."""
        if not HAS_PANDAS:
            raise ImportError("pandas is required")
        data: Dict[str, np.ndarray] = {"DEPTH": self.depth}
        if self.grey_level is not None:
            data["GREY"] = self.grey_level
        if self.grey_std is not None:
            data["GREY_STD"] = self.grey_std
        if self.r is not None:
            data["R"] = self.r
        if self.g is not None:
            data["G"] = self.g
        if self.b is not None:
            data["B"] = self.b
        if self.hue is not None:
            data["HUE"] = self.hue
        if self.saturation is not None:
            data["SAT"] = self.saturation
        if self.value is not None:
            data["VAL"] = self.value
        data["BRIGHTNESS"] = self.brightness_index()
        data["COLOR_INDEX"] = self.color_index()
        return pd.DataFrame(data)

    def __repr__(self) -> str:
        channels = []
        if self.grey_level is not None:
            channels.append("grey")
        if self.has_rgb:
            channels.append("RGB")
        if self.has_hsv:
            channels.append("HSV")
        if self.grey_std is not None:
            channels.append("texture")
        depth_text = (
            f"{self.top:.1f}-{self.bottom:.1f}"
            if self.top is not None and self.bottom is not None
            else "None-None"
        )
        return (f"CoreImageData(samples={self.n_samples}, "
                f"depth={depth_text}, "
                f"channels=[{', '.join(channels) or 'none'}])")
