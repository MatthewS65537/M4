import sys
sys.path.append("./models")

from FCN import *
from EEGEncoder import *
from MMMM import *
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

def INITIALIZE_MODEL(device="cuda"):
	eeg_enc = EEGEncoder(
		enc_feat=1024,
		dec_emb_sz=768,
		enc_nhead=8,
		enc_dim_ff=2048,
		num_enc_layers=8,
		device=device
		)

	eeg_enc.add_head(
		name="EEG-TXT-BART",
		head=FCN(
			input_dim=840,
			output_dim=1024,
			hidden_dim=1024,
			num_layers=2,
			device=device
			)
		)

	eeg_enc.add_head(
		name="EEG-IMG-BRAIN2IMAGE",
		head=FCN(
			input_dim=128,
			output_dim=1024,
			hidden_dim=1024,
			num_layers=2,
			device=device
			)
		)

	model = MMMM(
		eeg_encoder=eeg_enc,
		device=device
		)

	BART_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
	BART_pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

	MMMM.add_branch(
		name="EEG-TXT-BART",
		branch=Branch(
			head=FCN(
				input_dim=768,
				output_dim=1024,
				num_layers=2,
				device=device
				)
			),
			body=BART_pretrained
			device=device
		)

	return MMMM