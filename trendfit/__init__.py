from .options import set_options
from ._version import get_versions


__all__ = ['models', 'bootstrap', 'stats']


__version__ = get_versions()['version']
del get_versions
