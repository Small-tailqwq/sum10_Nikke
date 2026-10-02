"""Solver package; importing it does not initialize OCR or input devices."""
import sys as _sys

# Script-mode imports register a package alias so Numba caches work both ways.
# If that happened before this package was imported, preserve normal attribute
# access for ``import Head.certified_solver; Head.certified_solver.solve(...)``.
_cached_solver = _sys.modules.get(__name__ + '.certified_solver')
if _cached_solver is not None:
    certified_solver = _cached_solver
