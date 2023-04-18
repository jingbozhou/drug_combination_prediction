import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M:%S',
    stream=sys.stderr)

# Load all modules
from .data import *
from .features import *
from .prediction import *
from .MultiProcessing import *
#from .main import *
