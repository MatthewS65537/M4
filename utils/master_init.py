import sys
sys.path.append("./models")
sys.path.append("./utils")

from FCN import *
from EEGEncoder import *
from DiffusionHead import *
from ClassificationHead import *
from MMMM import *
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig, CLIPTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel, LMSDiscreteScheduler
from load_data import *
from dataloader import *
import torch

def INITIALIZE_MODEL(device="cuda", device_ids=None, dtype=torch.float32):
    """
    Initializes the model with the specified device and device IDs.

    Args:
        device (str, optional): The device to use for the model. Defaults to "cuda".
        device_ids (list[int], optional): The IDs of the devices to use for parallel processing. Defaults to None.

    Returns:
        MMMM: The initialized model.

    Raises:
        None
    """
#     torch.set_default_dtype(torch.float16)
    
    ### SET UP EEG ENCODER###
    eeg_enc = EEGEncoder(
        enc_feat=1024,
        dec_emb_sz=768,
        enc_nhead=8,
        enc_dim_ff=2048,
        num_enc_layers=8,
        device=device,
        dtype=dtype
        )

    eeg_enc.add_head(
        name="EEG-TEXT-BART",
        head=FCN(
            input_dim=840,
            output_dim=1024,
            hidden_dim=1024,
            num_layers=1,
            device=device,
            dtype=dtype
            )
        )

    eeg_enc.add_head(
        name="EEG-IMG-BRAIN2IMAGE",
        head=FCN(
            input_dim=128,
            output_dim=1024,
            hidden_dim=1024,
            num_layers=1,
            device=device,
            dtype=dtype
            )
        )

    CLIPtext_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(dtype=dtype)
    # image_encoder = CLIPImageModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        
    ### CREATE MMMM MODEL ###
    model = MMMM(
        eeg_encoder=eeg_enc,
        CLIP_text_encoder = CLIPtext_encoder,
        BART_text_encoder = None,
        image_encoder = None,
        dual_encoder = None,
        fusion_encoder = None,
        device = device,
        device_ids = device_ids,
        dtype = dtype
        )

    ### LOAD EEG-TEXT-BART ###
    BART_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    BART_pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large').to(dtype=dtype)
    
#     if not device_ids == None:
#         BART_pretrained = nn.DistributedDataParallel(BART_pretrained, device_ids=device_ids)

    # Create Branch for EEG-TEXT-BART
    BART_branch = Branch(
        head=FCN(
            input_dim=768,
            output_dim=1024,
            num_layers=1,
            device=device
            ),
        body=BART_pretrained,
        device=device,
        dtype=dtype
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
            device=device,
            dtype=dtype
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
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", torch_dtype=dtype)
    unet.requires_grad_(False)

    # Initializing CLIP Pretrains
    CLIPtokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    # CLIPtext_encoder is already initialized
    # CLIPtext_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

    # Create Branch for EEG-IMG-DIFFUSION
    DIFFUSION_branch = Branch(
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
        device=device,
        dtype=dtype
        )

    # Adding Branch
    model.add_branch(
        name="EEG-IMG-DIFFUSION",
        branch=DIFFUSION_branch,
        )

    # Adding Head
    model.add_head(
        name="EEG-IMG-DIFFUSION",
        head=FCN(
            input_dim=768,
            output_dim=768,
            num_layers=4,
            device=device,
            dtype=dtype
            )
    )

    ### LOAD CLASSIFICATION HEADS ###
    IMAGENET_CLASSIFICATION_branch = Branch(
        head=FCN(
            input_dim=768,
            output_dim=512,
            num_layers=1,
            device=device,
            dtype=dtype
            ),
        body=ClassificationHead(
            input_dim=512,
            output_dim=40,
            hidden_dim=1024,
            num_layers=4,
            device=device,
            dtype=dtype
            ),
        device=device,
        dtype=dtype
    )

    # Adding Branch
    model.add_branch(
        name="EEG-IMG-BRAIN2IMAGE-CLASSIFICATION",
        branch=IMAGENET_CLASSIFICATION_branch
        )
    
    ## Adding Head
    model.add_head(
        name="EEG-IMG-BRAIN2IMAGE-CLASSIFICATION",
        head=FCN(
            input_dim=768,
            output_dim=768,
            num_layers=4,
            device=device,
            dtype=dtype
            )
        )
    
    SENTIMENT_branch = Branch(
        head=FCN(
            input_dim=768,
            output_dim=512,
            num_layers=1,
            device=device,
            dtype=dtype
            ),
        body=ClassificationHead(
            input_dim=512,
            output_dim=3,
            hidden_dim=1024,
            num_layers=4,
            device=device,
            dtype=dtype
            ),
        device=device,
        dtype=dtype
    )
    model.add_branch(
        name="EEG-TEXT-BART-SENTIMENT",
        branch=SENTIMENT_branch
    )
    model.add_head(
        name="EEG-TEXT-BART-SENTIMENT",
        head=FCN(
            input_dim=768,
            output_dim=768,
            num_layers=4,
            device=device,
            dtype=dtype
        )
    )
    
    return model

def INITIALIZE_DATALOADERS(keys, bsz, dev_bsz=None):
    """
    Initialize and return a dictionary of dataloaders based on the given keys and batch sizes.

    Parameters:
    - keys (list): A list of keys representing the different datasets.
    - bsz (list): A list of batch sizes corresponding to each dataset.

    Returns:
    - dataloader_dict (dict): A dictionary containing the initialized dataloaders for each dataset.
    """
    dataloader_dict = {}
    idx = 0
    for key in keys:
        if key == "ZuCo-BART":
            ZuCo_data = load_txt_data("./data/ZuCo", "BART")
            master_eeg, master_embeds = ZuCo_data["data"], ZuCo_data["targets"]
            del ZuCo_data
            ZuCo_dataloader = {
                "train" : ZuCoDataloader(master_eeg["train"], master_embeds["train"], bsz=bsz[idx], drop_last=True),
                "dev" : ZuCoDataloader(master_eeg["dev"], master_embeds["dev"], bsz=1 if dev_bsz[idx] is None else dev_bsz[idx], drop_last=True),
                "test" : ZuCoDataloader(master_eeg["test"], master_embeds["test"], bsz=1 if dev_bsz[idx] is None else dev_bsz[idx], drop_last=True)
            }
            del master_eeg, master_embeds
            dataloader_dict[key] = ZuCo_dataloader
            del ZuCo_dataloader
        elif key == "ZuCo-CLIP":
            ZuCo_data = load_txt_data("./data/ZuCo", "CLIP")
            master_eeg, master_embeds = ZuCo_data["data"], ZuCo_data["targets"]
            del ZuCo_data
            ZuCo_dataloader = {
                "train" : ZuCoDataloader(master_eeg["train"], master_embeds["train"], bsz=bsz[idx], drop_last=True),
                "dev" : ZuCoDataloader(master_eeg["dev"], master_embeds["dev"], bsz=1 if dev_bsz[idx] is None else dev_bsz[idx], drop_last=True),
                "test" : ZuCoDataloader(master_eeg["test"], master_embeds["test"], bsz=1 if dev_bsz[idx] is None else dev_bsz[idx], drop_last=True)
            }
            del master_eeg, master_embeds
            dataloader_dict[key] = ZuCo_dataloader
            del ZuCo_dataloader
        elif key == "Brain2Image":
            Brain2Image_data = load_img_data("./data/Brain2Image")
            labels, img_net_dict, label_dict = Brain2Image_data["data"], Brain2Image_data["targets"], Brain2Image_data["labels"]
            del Brain2Image_data
            if not bsz[idx] == 1:
                print("[WARNING] Batch Size for Brain2Image AKA ImageNet is NOT 1. UNEXPECTED BEHAVIOR MAY OCCUR.")
            Brain2Image_dataloader = {
                "train": ImageNetDataloader(labels["train"], img_net_dict, label_dict, bsz=bsz[idx], drop_last=True),
                "dev": ImageNetDataloader(labels["dev"], img_net_dict, label_dict, bsz=1 if dev_bsz[idx] is None else dev_bsz[idx], drop_last=True),
                "test": ImageNetDataloader(labels["test"], img_net_dict, label_dict, bsz=1 if dev_bsz[idx] is None else dev_bsz[idx], drop_last=True)
            }
            del labels, img_net_dict
            dataloader_dict[key] = Brain2Image_dataloader
            del Brain2Image_dataloader
        idx += 1
    return dataloader_dict