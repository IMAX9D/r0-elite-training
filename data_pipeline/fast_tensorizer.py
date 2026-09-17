"""Packed-bank feature filling: an equivalent, cheaper replacement for the reference
`training_r0.elite_tensorizer.fill`.

Why this exists
---------------
The reference fill writes one field at a time:

    for j, key in enumerate(names):
        if key in values:
            destination[2*j:2*j+2] = torch.tensor([values[key], 1.])

Weighted py-spy sampling on the real EPYC production line puts this function's subtree at
21.1% of producer time and its self time at 21.0% -- the single largest hot spot, ahead of
`_actor` (11.1%). (An earlier attempt to rank it counted distinct stacks instead of sample
weights and wrongly reported 1.38%; that number was withdrawn.) The reference also walks
all `names` -- CHILD_FEATURES alone is 72 entries -- on every call.

Contract safety
---------------
`current_contract()` hashes `training_r0/*.py`, so editing that tree would change
`contract_sha256` and orphan the ~70 GB already compiled under `035a9c24...`. This module
therefore **never edits `training_r0`**; it installs an equivalent function at runtime,
selected explicitly by `r0_compile.py` (`select_compile_backend`). The reference stays
importable and untouched as the oracle for A/B comparison.

Equivalence requirements (review section 2.3)
---------------------------------------------
* Same field order, dtype, shape and write set.
* `(0, 0)` unknown vs `(0, 1)` known-zero must stay distinguishable: only keys present in
  `values` are written, and the known bit is written as exactly 1.0.
* Cells for absent keys are **left untouched**. This is deliberately NOT a whole-bank
  overwrite -- the reference only writes the cells it hits, so a caller that fills the same
  destination twice would see different results if we overwrote.
* No fast-math and no dtype change: values go through the same float64 -> float32 store the
  reference gets from `torch.tensor([...])`.

Known precondition: **duplicate names**. The reference loops over positions, so a name that
appears twice is written at BOTH positions. A `name -> j` dict would keep only the last one
and silently drop the first. `_index_for` therefore detects duplicates and this module falls
back to a verbatim copy of the reference for that field list rather than guessing.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = ['fill', 'install', 'reference_fill', 'is_installed', 'verify_preconditions']

# name tuple -> (names, {name: j}, ok). The `names` arguments are module-level frozen tuples
# (CHILD_FEATURES, GROUP_FEATURES, ...), so identity is stable and the index is built once
# per tuple instead of re-enumerating up to 72 names on every call.
_INDEX_CACHE: dict = {}
_ORIGINAL = None
_FALLBACKS = 0
_INSTALLED_AT = None


def reference_fill(destination, names, values) -> None:
    """Verbatim copy of the reference implementation.

    Used whenever the packed path's preconditions do not hold. Deliberately byte-for-byte
    the same logic as `training_r0.elite_tensorizer.fill` so the fallback cannot be a
    "close enough" approximation that quietly changes results.
    """
    for j, key in enumerate(names):
        if key in values:
            destination[2 * j:2 * j + 2] = torch.tensor([values[key], 1.])


def _index_for(names):
    """Return (index, ok). ok=False means the packed path must not handle this field list."""
    cached = _INDEX_CACHE.get(id(names))
    if cached is not None and cached[0] is names:
        return cached[1], cached[2]
    index: dict = {}
    ok = True
    for j, name in enumerate(names):
        if name in index:
            ok = False     # duplicate: dict would keep only the last position
            break
        index[name] = j
    _INDEX_CACHE[id(names)] = (names, index, ok)
    return index, ok


def fill(destination, names, values) -> None:
    """Drop-in replacement for `training_r0.elite_tensorizer.fill`."""
    global _FALLBACKS
    if not values:
        return
    # Preconditions. Cheap attribute checks only; a failure routes to the verbatim
    # reference rather than raising, because aborting a 2000-battle batch over one
    # unexpected field list would be worse than being slow.
    if destination.dtype is not torch.float32 or destination.device.type != 'cpu':
        _FALLBACKS += 1
        return reference_fill(destination, names, values)
    index, ok = _index_for(names)
    if not ok:
        _FALLBACKS += 1
        return reference_fill(destination, names, values)
    # `destination` here is the trailing slice of a freshly zeroed CPU float32 bank, so
    # `.numpy()` shares storage. (It is not true in general that a non-contiguous tensor
    # cannot be converted; for this call pattern the slice is contiguous, and if a future
    # caller passes something else the exception path falls back to the reference.)
    try:
        view = destination.numpy()
    except (RuntimeError, TypeError):
        _FALLBACKS += 1
        return reference_fill(destination, names, values)
    for key, value in values.items():
        j = index.get(key)
        if j is not None:
            view[2 * j] = value
            view[2 * j + 1] = 1.0


def verify_preconditions(verbose: bool = False) -> dict:
    """Check the assumptions the packed path relies on, at process start.

    Verifies that every field list the compiler uses has unique names. Returns a report so
    the caller can refuse to install the backend instead of running it on inputs it was
    never validated against.
    """
    from training_r0.feature_contract import (CHILD_FEATURES, TOWER_FEATURES,
                                              GROUP_FEATURES, CARD_FEATURES, MATCH_FEATURES,
                                              COMBAT_FEATURES, ABILITY_FEATURES)
    groups = dict(CHILD_FEATURES=CHILD_FEATURES, TOWER_FEATURES=TOWER_FEATURES,
                  GROUP_FEATURES=GROUP_FEATURES, CARD_FEATURES=CARD_FEATURES,
                  MATCH_FEATURES=MATCH_FEATURES, COMBAT_FEATURES=COMBAT_FEATURES,
                  ABILITY_FEATURES=ABILITY_FEATURES)
    duplicates = {}
    for name, names in groups.items():
        seen, dupes = set(), []
        for field in names:
            if field in seen:
                dupes.append(field)
            seen.add(field)
        if dupes:
            duplicates[name] = dupes
    report = dict(field_lists={k: len(v) for k, v in groups.items()},
                  duplicates=duplicates,
                  unique=not duplicates,
                  dtype_ok=torch.zeros(1).dtype is torch.float32)
    if verbose:
        print(f'fast_tensorizer preconditions: unique={report["unique"]} '
              f'default_dtype_float32={report["dtype_ok"]} '
              f'lists={report["field_lists"]}', flush=True)
        if duplicates:
            print(f'  !! duplicate feature names: {duplicates}', flush=True)
    return report


def install(require_unique: bool = True) -> bool:
    """Point `training_r0.elite_tensorizer.fill` at the packed implementation.

    `encode_elite` resolves `fill` through its module globals at call time, so replacing the
    module attribute is enough -- no `training_r0` source file is touched.

    Intended to be called ONCE, at compile-process start. It is not designed to be toggled
    between concurrent tasks: `_ORIGINAL` captures whatever was installed first, so
    interleaved install/uninstall across threads would restore the wrong function.
    """
    global _ORIGINAL, _INSTALLED_AT
    import time
    report = verify_preconditions()
    if require_unique and not report['unique']:
        print('fast_tensorizer: NOT installing -- duplicate feature names would be '
              'silently mis-written by the packed path', flush=True)
        return False
    from training_r0 import elite_tensorizer
    if _ORIGINAL is None:
        _ORIGINAL = elite_tensorizer.fill
    elite_tensorizer.fill = fill
    _INSTALLED_AT = time.time()
    return True


def uninstall() -> bool:
    """Restore the reference fill. Only for the A/B harness, never during production."""
    global _ORIGINAL
    if _ORIGINAL is None:
        return False
    from training_r0 import elite_tensorizer
    elite_tensorizer.fill = _ORIGINAL
    return True


def is_installed() -> bool:
    from training_r0 import elite_tensorizer
    return elite_tensorizer.fill is fill


def backend_identity() -> dict:
    """Identity of the active backend, for recording in the compiled manifest.

    The review requires this: keeping the old reference's contract hash does not mean the
    new execution path has the same implementation identity, so each artifact must say
    which backend produced it.
    """
    import hashlib
    from pathlib import Path
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return dict(compiler_backend='packed' if is_installed() else 'reference',
                backend_build_sha256=digest,
                equivalence_suite='fill_ab.py:15-captures/1059388-tensors/bit-identical',
                fallbacks=_FALLBACKS,
                cache_entries=len(_INDEX_CACHE),
                installed_at=_INSTALLED_AT)
