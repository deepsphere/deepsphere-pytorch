"""DeepSphere Base Documentation doc
"""

import importlib
import sys

__version__ = "0.2.1"


def import_modules(names, src, dst):
    """Import modules in package."""
    for name in names:
        module = importlib.import_module("{}.{}".format(src, name))
        setattr(sys.modules[dst], name, module)


__all__ = []

import_modules(__all__[::-1], "deepsphere", "deepsphere")
