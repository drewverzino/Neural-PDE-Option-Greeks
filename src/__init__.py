"""Top-level package for the Neural-PDE-Option-Greeks project.

This module exposes common filesystem paths so submodules can rely on
package-relative locations instead of the caller's working directory.
"""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure expected directories exist when the package is imported.
for _path in (DATA_DIR, FIGURES_DIR, RESULTS_DIR):
    _path.mkdir(parents=True, exist_ok=True)

__all__ = [
    "PACKAGE_ROOT",
    "PROJECT_ROOT",
    "DATA_DIR",
    "FIGURES_DIR",
    "RESULTS_DIR",
]
