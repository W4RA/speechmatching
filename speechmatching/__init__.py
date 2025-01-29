"""Initialization of the speechmatching package.

Often used classes and functions are imported here to be able to be easily
found and imported.
"""

__version__ = '1.0.0'
__author__ = 'Auke Schuringa <auke.schuringa@proton.me>'
__maintainer__ = 'Auke Schuringa'
__email__ = 'auke.schuringa@proton.me'
__homepage__ = 'https://github.com/W4RA/speechmatching'
__description__ = 'Library for matching spoken words.'

from speechmatching.model import Transcriptor
from speechmatching.recording import (
    Group,
    Recording,
    load_directory,
    load_directory_groups
)

