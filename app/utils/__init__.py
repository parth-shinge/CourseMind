"""app.utils package.

Re-exports the shared helpers from the top-level utils.py module so that all
existing ``from app.utils import sanitize_lecture_id`` imports continue to
work after the utils/ package was added in Phase 6.
"""

from __future__ import annotations

# Re-export from the sibling utils.py file (which still lives at app/utils.py).
# We import via importlib to avoid the name collision between this package and
# the .py file — Python 3 resolves "app.utils" to this package (directory),
# so we need to reach the .py file explicitly.

import importlib.util
import sys
from pathlib import Path

_utils_py = Path(__file__).parent.parent / "utils.py"  # app/utils.py
_spec = importlib.util.spec_from_file_location("_app_utils_py", _utils_py)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

sanitize_lecture_id = _mod.sanitize_lecture_id

__all__ = ["sanitize_lecture_id"]
