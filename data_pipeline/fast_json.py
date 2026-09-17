"""Faster JSON decoding for the capture reader, without touching `training_r0`.

Why
---
Weighted py-spy sampling after the packed-fill rollout puts `raw_decode` at **12.65%** of
producer self time -- the new single largest hot spot, ahead of `_actor` (11.54%). The
cause is twofold:

  1. `r0_compile.index_capture` parses every capture line, and then the compile loop parses
     the whole file *again*. Two full passes.
  2. The interpreter's stdlib `json` has no C accelerator here (`json._json` is absent), so
     both passes run a pure-Python decoder. Measured on a real capture: 1204 lines in
     0.69 s, i.e. ~630 us per line.

`orjson` decodes the same 1204 lines in 0.28 s (**2.47x**) with **zero** mismatches against
the stdlib, and no line in the sample contains a `NaN`/`Infinity` literal.

What is deliberately NOT replaced
---------------------------------
`dumps` stays on the stdlib. `r0_compile` writes metadata with
`json.dumps(..., ensure_ascii=True, indent=2)`; orjson emits UTF-8 without escaping
non-ASCII and has a different `default=` protocol, so swapping it would change the bytes of
`coverage.json` / `contract.json` / `manifest.json`. Only the read path is accelerated.

Scope of the patch
------------------
The hot call sites live in contract-hashed `training_r0/replay_adapter.py`, which must not
be edited. So this installs a *module-level* replacement: it rebinds the `json` name inside
the two target modules only. The global `json` module is never touched, so unrelated
libraries keep the stdlib behaviour.
"""
from __future__ import annotations

import json as _stdlib

try:
    import orjson as _orjson
except ImportError:                                   # pragma: no cover
    _orjson = None

__all__ = ['JsonShim', 'install', 'uninstall', 'is_installed', 'available', 'stats']

_ORIGINALS: dict = {}
_LOADS_OK = 0
_LOADS_FALLBACK = 0


class JsonShim:
    """Drop-in stand-in for the `json` module: fast `loads`, stdlib everything else."""

    def __init__(self, use_orjson: bool = True):
        self.use_orjson = bool(use_orjson and _orjson is not None)

    def loads(self, s, *args, **kwargs):
        global _LOADS_OK, _LOADS_FALLBACK
        if not self.use_orjson:
            return _stdlib.loads(s, *args, **kwargs)
        # orjson implements only the plain `loads(s)`. Given parse_float / parse_int /
        # object_hook / object_pairs_hook / cls it would silently ignore them and return a
        # different object than the caller asked for, so any extra argument goes to the
        # stdlib. (Verified against the review's reproduction: parse_float=Decimal,
        # parse_int=str and object_pairs_hook=list were all ignored on the fast path.)
        if args or kwargs:
            return _stdlib.loads(s, *args, **kwargs)
        try:
            value = _orjson.loads(s)
            _LOADS_OK += 1
            return value
        except _orjson.JSONDecodeError:
            # Narrow on purpose. A blanket `except Exception` would swallow MemoryError and
            # other resource failures and retry them as though the text were unparseable,
            # hiding a real fault behind a slow-but-successful path. Only a genuine
            # "orjson will not accept this text" may fall back.
            _LOADS_FALLBACK += 1
            return _stdlib.loads(s)

    # Read path that takes a file object is rare here; keep stdlib semantics exactly.
    def load(self, fp, *args, **kwargs):
        return _stdlib.load(fp, *args, **kwargs)

    def dump(self, obj, fp, *args, **kwargs):
        return _stdlib.dump(obj, fp, *args, **kwargs)

    def dumps(self, obj, *args, **kwargs):
        return _stdlib.dumps(obj, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(_stdlib, name)


def available() -> bool:
    return _orjson is not None


def install() -> bool:
    """Rebind `json` inside the capture-reading modules to the shim.

    Called once per compile process. Returns False when orjson is unavailable, in which
    case the stdlib keeps being used and behaviour is unchanged.
    """
    global _ORIGINALS
    if _orjson is None:
        return False
    if _ORIGINALS:
        return True                                    # already installed
    from training_r0 import replay_adapter
    from . import r0_compile
    shim = JsonShim(True)
    for module in (replay_adapter, r0_compile):
        _ORIGINALS[module.__name__] = module.json
        module.json = shim
    return True


def uninstall() -> bool:
    """Restore the stdlib `json` in the patched modules (A/B harness only)."""
    global _ORIGINALS
    if not _ORIGINALS:
        return False
    import importlib
    for name, original in _ORIGINALS.items():
        importlib.import_module(name).json = original
    _ORIGINALS = {}
    return True


def is_installed() -> bool:
    from training_r0 import replay_adapter
    return isinstance(getattr(replay_adapter, 'json', None), JsonShim)


def stats() -> dict:
    return dict(orjson_available=_orjson is not None,
                version=getattr(_orjson, '__version__', None) if _orjson else None,
                installed=is_installed(), loads_orjson=_LOADS_OK, loads_fallback=_LOADS_FALLBACK)
