"""Project configurations such as logger, color scheme etc"""

import logging
from itertools import cycle

# logger configuration
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y %H:%M:%S',
    level=logging.INFO,
    # filename='logs.txt'
)

# color schemes
rgb = {
    'green_main': (112, 182, 88),
    'green_dark': (33, 84, 37),
    'grey_dark': (49, 45, 49),
    'dog': (191, 209, 67),
    'cat': (232, 132, 65),
    'small_pet': (212, 153, 59),
    'fish': (40, 58, 140),
    'bird': (109, 173, 218),
    'reptile': (101, 38, 57),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
}

opacity: float = 0.6
rgba = {k: ('rgba' + str(v)[:-1] + f', {opacity})') for k, v in rgb.items()}
rgba_vals = list(rgba.values())
rgba_inf = cycle(rgba.values())

