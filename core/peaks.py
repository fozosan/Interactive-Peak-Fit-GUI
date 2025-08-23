"""Peak representation for Peakfit 3.x."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, List


@dataclass
class Peak:
    """Simple container describing a pseudo-Voigt peak."""

    center: float
    height: float
    fwhm: float
    eta: float
    lock_center: bool = False
    lock_width: bool = False


def serialize(peaks: Iterable[Peak]) -> List[dict]:
    """Serialize ``peaks`` into dictionaries suitable for JSON/CSV export."""

    return [asdict(p) for p in peaks]


def deserialize(records: Iterable[dict]) -> List[Peak]:
    """Reconstruct ``Peak`` objects from an iterable of dictionaries."""

    return [Peak(**rec) for rec in records]
