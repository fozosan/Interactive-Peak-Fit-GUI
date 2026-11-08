# Performance Apply Debounce Pattern - Knowledge Base

## Problem Solved

**Issue:** Repeated identical backend log spam during app startup and field editing:
```
Backend: numpy | fit_workers_req=0 ... seed_all=False seed=-
Backend: numpy | fit_workers_req=0 ... seed_all=False seed=-
... (dozens of times)
```

**Root Cause:** Tk variable traces (`trace_add("write", ...)`) fire multiple times per edit cycle, causing `apply_performance()` to execute repeatedly.

## Solution Pattern Implemented

### 1. Scheduler/Debounce Infrastructure

Add these fields to the class:
```python
self._perf_apply_after_id: Optional[int] = None      # Tk after() id for pending apply
self._perf_last_signature: Optional[tuple] = None    # Last effective state signature  
self._perf_pending_source: str = ""                  # Source of last scheduled call
```

### 2. Request Method

`request_apply_performance(source, immediate=False, save_cfg=True, force_log=False)`:

- **Cancel pending** apply if one exists (via `root.after_cancel()`)
- If `immediate=True`: run now without debounce  
- If `immediate=False`: schedule with 150ms debounce window
- Debounced calls collapse rapid traces into single apply

### 3. Signature Computation

`_compute_perf_signature()` returns tuple of:
```python
(backend_name, fit_req, unc_req, strategy, 
 mkl_threads, openblas_threads, omp_threads, seed_all, seed_val)
```

Used to detect effective state changes for log deduplication.

### 4. Core Apply with Logging Dedupe

`_apply_performance_now(source, save_cfg=True, force_log=False)`:

1. Compute new signature
2. Compare with last signature → `sig_changed = (new != old)`
3. Log only if: `force_log OR sig_changed`
4. If skipping duplicate: log debug message instead
5. Save config only if `save_cfg=True`

### 5. Trigger Wiring

**Trace bindings** (typing, changes):
```python
self.perf_numba.trace_add("write", 
    lambda *_: self.request_apply_performance(source="trace:perf_numba"))
# immediate=False by default → debounced
# save_cfg=False by default → no disk I/O spam
```

**Explicit actions** (Apply button, radio clicks):
```python
ttk.Button(..., command=self._apply_performance_immediate)
# Runs immediately with logging always enabled
```

## Benefits Achieved

1. **Reduced log spam**: Only logs on actual state changes or user action
2. **Less disk I/O**: Config save only on explicit Apply
3. **Better UX**: Rapid edits don't flood console
4. **Backward compatible**: Existing `apply_performance()` still works

## Common Pitfalls to Avoid

1. **Never use direct apply_performance() in trace bindings** - always use request_apply_performance() with appropriate parameters
2. **Ensure signature computation is robust** - handle missing attributes gracefully (use hasattr checks)
3. **Don't forget to update Apply button** - must call immediate path for responsive UX
4. **Track after() id properly** - cancel pending on new requests to prevent duplicate execution

## Testing Checklist

- [ ] Startup produces ≤1 backend log line
- [ ] Typing in seed field doesn't spam logs  
- [ ] Clicking Apply shows exactly one backend log
- [ ] Changing strategy (outer/inner) shows one backend log per change
- [ ] Config persists only on explicit Apply, not traces


## Critical Lesson Learned

**Attribute Initialization Order Matters:**

When adding new instance attributes that will be used during `__init__()`, ensure they're defined BEFORE the first use. In this case:

WRONG order (caused AttributeError):
```python
self._build_ui()
self.request_apply_performance(...)  # Tries to access _perf_apply_after_id - DOESN'T EXIST YET!
# ... later ...
self._perf_apply_after_id = None     # Too late! AttributeError!
```

CORRECT order:
```python
self._build_ui()
# Define scheduler fields FIRST
self._perf_apply_after_id = None
self._perf_last_signature = None  
self._perf_pending_source = ""
# NOW they exist when needed
self.request_apply_performance(...)  # Safe to use!
```

**Rule:** Always define instance attributes before any method that accesses them, even if those methods are called later in the same `__init__()`.
