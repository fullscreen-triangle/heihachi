"""
Heihachi - Neural Processing of Electronic Music

A high-performance audio analysis framework designed for electronic music,
with a particular focus on neurofunk and drum & bass genres.
"""

__version__ = "0.1.0"
__author__ = "Kundai Sachikonye"

# Import main components for easier access
# These will make core functionality available directly from the package

try:
    # Attempt to import core components - these may fail during installation
    # but should work after the package is fully installed
    from . import core
    from . import alignment
    from . import annotation
    from . import utils
    from . import visualizations
    from . import feature_extraction
except ImportError:
    # Imports may fail during installation, which is expected
    pass
