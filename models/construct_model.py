import sys
sys.path.append("./models")

from FCN import *
from EEGEncoder import *
from MMMM import *

def construct_model(device=""):
	return model