import numpy as np

COLOR_BLACK     = np.array([0, 0, 0])
COLOR_WHITE     = np.array([255, 255, 255])
COLOR_RED       = np.array([255, 0, 0])
COLOR_GREEN     = np.array([0, 255, 0])
COLOR_BLUE      = np.array([0, 0, 255])
COLOR_YELLOW    = np.array([255, 255, 0])
COLOR_MAGENTA   = np.array([255, 0, 255])
COLOR_CYAN      = np.array([0, 255, 255])
# color palette
PALLETTE_BASIC = np.array([
    COLOR_BLACK,    # black
    COLOR_WHITE,    # white
    COLOR_RED,      # red
    COLOR_GREEN,    # green
    COLOR_BLUE,     # blue
    COLOR_YELLOW,   # yellow
    COLOR_MAGENTA,  # magenta
    COLOR_CYAN      # cyan
])

# big palette
PALLETTE = np.array([
    [0, 0, 0],      # black
    [0, 0, 128],    # dark blue
    [0, 128, 0],    # dark green
    [0, 128, 128],  # dark cyan
    [128, 0, 0],    # dark red
    [128, 0, 128],  # dark magenta
    [128, 128, 0],  # dark yellow
    [192, 192, 192],# light gray
    [128, 128, 128],# dark gray
    [0, 0, 255],    # blue
    [0, 255, 0],    # green
    [0, 255, 255],  # cyan
    [255, 0, 0],    # red
    [255, 0, 255],  # magenta
    [255, 255, 0],  # yellow
    [255, 255, 255] # white
])

# palette with very light colors
PALLETTE_PASTEL = np.array([
    # [0, 0, 0],      # black
    # [30, 144, 255],    # dodger blue
    # [0, 255, 127],    # spring green
    # [0, 255, 255],  # cyan
    # [255, 0, 0],    # red
    # [255, 0, 255],  # magenta
    # [255, 255, 0],  # yellow
    # [255, 255, 255], # white
    # 16 colors
    [255, 255, 240], # ivory
    [240, 255, 240], # honeydew
    [240, 255, 255], # azure
    [240, 240, 255], # alice blue
    [255, 240, 245], # lavender blush
    [255, 218, 185], # peach puff
    [255, 222, 173], # navajo white
    [255, 228, 181], # moccasin
    [255, 239, 213], # papaya whip
    [255, 245, 238], # seashell
    [245, 245, 220], # beige
    [253, 245, 230], # old lace
    [250, 235, 215], # antique white
    [250, 240, 230], # linen
    [255, 250, 240], # floral white
    [255, 250, 250], # snow
])

from .FISTAT6 import *
from .SHAG_CARPET import *
from .PIXLS_DEFAULT import *
from .PAITO24 import *
from .PAPER10 import *
from .ROSE_BUS import *
from .AZURE_ABYSS import *
from .EXOPHOBIA import *
from .ANSI_FORLORN_64 import *
from .AROACIDIC import *
from .A_BIZARRE_CONCOTION import *
from .LOSPEC500 import *
from .MIDNIGHT_ABLAZ import *
from .P_1BIT_MONITOR_GLOW import *
from .RUBIKS_CUBE_COLORS import *
