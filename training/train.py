import sys

sys.path.append("./models")
sys.path.append("./training")
sys.path.append("./utils")

from eeg_encoder import *
from DSG import *
from load_data import load_txt_data, load_img_data

if __name__ == "__main__":
	load_txt_data()
	load_img_data()