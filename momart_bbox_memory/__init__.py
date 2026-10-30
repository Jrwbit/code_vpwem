"""MoMaRT main-view bounding-box interaction memory experiment."""

# Allow evaluation to load the RGB-only online detector package from the
# separate detector checkout while keeping policy modules in this checkout.
from pkgutil import extend_path
import os
from pathlib import Path

__path__ = extend_path(__path__, __name__)
precision_root = os.environ.get("PRECISION_CODE_ROOT")
if precision_root:
    precision_package = str(Path(precision_root) / __name__)
    if Path(precision_package).is_dir():
        __path__ = [precision_package] + [
            path for path in __path__ if path != precision_package]
