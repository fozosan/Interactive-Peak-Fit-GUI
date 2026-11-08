"""Peak representation for Peakfit 3.x."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Iterable, List, Optional


@dataclass
class Peak:
    """Simple container describing a pseudo-Voigt peak."""

    center: float
    height: float
    fwhm: float
    eta: float
    lock_center: bool = False
    lock_width: bool = False


_FIELDS = tuple(f.name for f in fields(Peak))


def serialize(peaks: Iterable[Peak]) -> List[dict]:
    """Serialize ``peaks`` into dictionaries suitable for JSON/CSV export."""

    return [asdict(p) for p in peaks]


def deserialize(records: Iterable[dict]) -> List[Peak]:
    """Reconstruct ``Peak`` objects from an iterable of dictionaries.

    The records may be the GUI's own peaks (``ui.app.Peak.__dict__``, which is
    also what a saved template holds). Those carry ``fwhm_cap_enabled`` and
    ``fwhm_cap_value``, which this class does not have; keys it does not have
    are dropped. Read the caps with :func:`width_caps`.
    """

    return [Peak(**{k: rec[k] for k in _FIELDS if k in rec}) for rec in records]


def width_caps(records: Iterable[dict]) -> List[Optional[float]]:
    """Return each record's FWHM cap, in the form solvers take as ``width_caps``.

    A cap is ``fwhm_cap_value`` when ``fwhm_cap_enabled`` is set and the width
    is not locked, and ``None`` otherwise: the rule the GUI applies to its own
    peaks when it builds solver options.
    """

    return [
        float(rec.get("fwhm_cap_value", float("inf")))
        if rec.get("fwhm_cap_enabled") and not rec.get("lock_width")
        else None
        for rec in records
    ]
