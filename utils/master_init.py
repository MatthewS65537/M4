import sys
sys.path.append("./models")
sys.path.append("./utils")

from FCN import *
from EEGEncoder import *
from DiffusionHead import *
from MMMM import *
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig, CLIPTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel, LMSDiscreteScheduler
from load_data import *
from dataloader import *

def INITIALIZE_MODEL(device="cuda", device_ids=None):
    ### SET UP EEG ENCODER###
    eeg_enc = EEGEncoder(
        enc_feat=1024,
        dec_emb_sz=768,
        enc_nhead=8,
        enc_dim_ff=2048,
        num_enc_layers=8,
        device=device,
        )

    eeg_enc.add_head(
        name="EEG-TEXT-BART",
        head=FCN(
            input_dim=840,
            output_dim=1024,
            hidden_dim=1024,
            num_layers=1,
            device=device
            )
        )

    eeg_enc.add_head(
        name="EEG-IMG-BRAIN2IMAGE",
        head=FCN(
            input_dim=128,
            output_dim=1024,
            hidden_dim=1024,
            num_layers=1,
            device=device
            )
        )

    ### CREATE MMMM MODEL ###
    model = MMMM(
        eeg_encoder=eeg_enc,
        device=device,
        )

    ### LOAD EEG-TEXT-BART ###
    BART_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    BART_pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    
    if not device_ids == None:
        BART_pretrained = nn.DataParallel(BART_pretrained, device_ids=device_ids)

    # Create Branch for EEG-TEXT-BART
    BART_branch = Branch(
        head=FCN(
            input_dim=768,
            output_dim=1024,
            num_layers=1,
            device=device
            ),
        body=BART_pretrained,
        device=device
        )

    # Adding Branch
    model.add_branch(
        name="EEG-TEXT-BART",
        branch=BART_branch
        )

    # Adding Head
    model.add_head(
        name="EEG-TEXT-BART",
        head=FCN(
            input_dim=768,
            output_dim=768,
            num_layers=4,
            device=device
        )
    )
    
    for name, param in BART_branch.named_parameters():
        if param.requires_grad and 'body' in name:
            if ('shared' in name) or ('embed_positions' in name) or ('encoder.layers.0' in name):
                continue
            else:
                param.requires_grad = False

    ### LOAD EEG-IMG-BRAIN2IMAGE ###
    
    # Initializing a scheduler and Setting number of sampling steps
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    scheduler.set_timesteps(50)
    
    # Initializing the U-Net model
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", torch_dtype=torch.float16).to(device)

    # Initializing CLIP Pretrains
    CLIPtokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    CLIPtext_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

    # Create Branch for EEG-IMG-BRAIN2IMAGE
    BRAIN2IMAGE_branch = Branch(
        head=FCN(
            input_dim=768,
            output_dim=768,
            num_layers=1,
            device=device
            ),
        body=DiffusionHead(
            scheduler=scheduler,
            unet=unet,
            tokenizer=CLIPtokenizer,
            text_encoder=CLIPtext_encoder,
            device=device
            ),
        device=device
        )

    # Adding Branch
    model.add_branch(
        name="EEG-IMG-BRAIN2IMAGE",
        branch=BRAIN2IMAGE_branch,
        )

    # Adding Head
    model.add_head(
        name="EEG-IMG-BRAIN2IMAGE",
        head=FCN(
            input_dim=768,
            output_dim=768,
            num_layers=4,
            device=device
            )
    )

    ### LOAD --- ###
    ## Dummy Code


    return model

def INITIALIZE_DATALOADERS(keys, bsz):
    dataloader_dict = {}
    idx = 0
    for key in keys:
        if key == "ZuCo-BART":
            ZuCo_data = load_txt_data("./data/ZuCo", "BART")
            master_eeg, master_embeds = ZuCo_data["data"], ZuCo_data["targets"]
            del ZuCo_data
            bsz=bsz[idx]
            ZuCo_dataloader = {
                "train" : ZuCoDataloader(master_eeg["train"], master_embeds["train"], bsz=bsz, drop_last=True),
                "dev" : ZuCoDataloader(master_eeg["dev"], master_embeds["dev"], bsz=1, drop_last=True),
                "test" : ZuCoDataloader(master_eeg["test"], master_embeds["test"], bsz=1, drop_last=True)
            }
            del master_eeg, master_embeds
            dataloader_dict[key] = ZuCo_dataloader
            del ZuCo_dataloader
        idx += 1
    return dataloader_dict
