import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionHead(nn.Module):
    def __init__(self, scheduler, unet, tokenizer, text_encoder, device):
        super(DiffusionHead, self).__init__()
        self.DiffusionHead = scheduler
        self.unet = unet
        self.device = device
        self.CLIPtokenizer = tokenizer
        self.CLIPtext_encoder = text_encoder

        self.to(device)

    def text_enc(self, prompts, maxlen=None):
        '''
        A function to take a textual promt and convert it into embeddings
        '''
        if maxlen is None: maxlen = self.CLIPtokenizer.model_max_length
        inp = self.CLIPtokenizer(prompts, padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt")
        return self.CLIPtext_encoder(inp.input_ids.to(self.device))[0].half()

    def forward(self, args_dict):
        '''
        g is guidance factor for diffusion model
        We likely won't change it, but leave it as an argument for now
        Maybe consider adding a dimension argument as well?
        '''
        emb=args_dict["input_data_batch"]
        steps=30
        g=0.5 # guidance fact
        bsz=emb.shape[0]
        uncond = self.text_enc([""] * bsz, emb.shape[1])
        emb = torch.cat([uncond, emb])
        self.scheduler.set_timesteps(steps)
        latents = latents.to(self.device).half() * self.scheduler.init_noise_sigma

        for i,ts in enumerate(self.scheduler.timesteps):
            # We need to scale the i/p latents to match the variance
            inp = self.scheduler.scale_model_input(torch.cat([latents] * 2), ts)

            # Predicting noise residual using U-Net
            u,t = self.unet(inp, ts, encoder_hidden_states=emb).sample.chunk(2)

            # Performing Guidance
            pred = u + g*(t-u)

            # Conditioning  the latents
            latents = self.scheduler.step(pred, ts, latents).prev_sample

        return latents