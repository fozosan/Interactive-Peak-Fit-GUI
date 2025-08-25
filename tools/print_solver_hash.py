#!/usr/bin/env python3
import inspect
import hashlib
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

modules = []
try:
    from fit import classic
    modules.append(classic)
    from fit import modern_vp
    modules.append(modern_vp)
    from fit import modern
    modules.append(modern)
    try:
        from fit import lmfit_backend
        modules.append(lmfit_backend)
    except Exception:
        pass
except Exception as exc:  # pragma: no cover - import errors shouldn't kill script
    print("import error", exc)

for mod in modules:
    try:
        src = inspect.getsource(mod)
        h = hashlib.sha256(src.encode("utf-8")).hexdigest()[:8]
        print(f"{mod.__name__} {h}")
    except Exception as e:  # pragma: no cover
        print(f"{mod.__name__} error {e}")
