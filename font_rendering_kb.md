# Tkinter Font Rendering / Antialiasing Improvements - Knowledge Base

## Issue: Missing Import Error in Type Annotations

### Problem Description
When creating `ui/fonts.py`, I used `ttk.Style` as a type hint but failed to import `ttk`. This caused a runtime `NameError`:
```
NameError: name 'ttk' is not defined. Did you mean: 'tk'?
```

### Root Cause
Type hints are evaluated at function definition time (not just when called). Using `style: ttk.Style` requires `ttk` to be available in the module's namespace at import time, even if the parameter isn't used until later.

### Solution Applied
Changed type hint from `style: ttk.Style` to `style: "ttk.Style"` (string annotation). This defers evaluation until runtime when the function is actually called.

```python
# Before (causes NameError):
def configure_ttk_styles(
    style: ttk.Style,  # ← Error if ttk not imported
    ...
) -> None:

# After (works with string annotation):
def configure_ttk_styles(
    style: "ttk.Style",  # ← String defers evaluation
    ...
) -> None:
```

### Best Practices for Type Annotations in Tkinter Modules

**Option 1: Use TYPE_CHECKING guard (cleanest)**
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tkinter import ttk

def configure_ttk_styles(
    style: "ttk.Style",  # Still need string annotation
    ...
) -> None:
```

**Option 2: Direct import (simplest)**
```python
from tkinter import tk, ttk

def configure_ttk_styles(
    style: ttk.Style,  # Works because ttk is imported
    ...
) -> None:
```

**Option 3: String annotation (safest for optional imports)**
```python
import tkinter as tk

def configure_ttk_styles(
    style: "ttk.Style",  # String defers evaluation
    ...
) -> None:
```

### When to Use Each Approach

| Approach | Best For | Pros | Cons |
|----------|----------|------|------|
| TYPE_CHECKING guard | Large projects, clean imports | IDE support, no runtime overhead | More verbose |
| Direct import | Small scripts, simple modules | Simple, readable | Imports at runtime |
| String annotation | Optional dependencies, circular refs | Flexible, works everywhere | Less IDE support in older Python |

### Lesson Learned

**Always verify that all types used in annotations are imported or use string annotations.** When using `ttk`, `tkinter.font`, or other submodules of tkinter:

1. Check existing imports first
2. If the type is from a submodule (e.g., `ttk.Style` not just `tk`), either:
   - Import it explicitly, OR  
   - Use string annotation `"submodule.Type"`

### Files Affected by This Pattern

- `ui/fonts.py` - Fixed with string annotation for `style: "ttk.Style"`
- Future files using similar patterns should follow the same approach
