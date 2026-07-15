from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class GntChar:
    tag_code: str
    width: int
    height: int
    bitmap: np.ndarray
    writer_code: Optional[str] = None


@dataclass
class DgrlLine:
    line_number: int
    image_width: int
    image_height: int
    label: str
    width: int
    height: int
    top: int
    left: int
    bitmap: np.ndarray
    writer_code: Optional[str] = None
