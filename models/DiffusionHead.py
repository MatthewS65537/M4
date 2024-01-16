import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionHead(nn.Module):
    def __init__(self, scheduler, unet, tokenizer, text_encoder, device=None, dtype=torch.float32):
        super(DiffusionHead, self).__init__()
#         OOPS!
        self.DiffusionHead = scheduler
        del self.DiffusionHead
        self.scheduler = scheduler
        self.unet = unet
        self.device = device
        self.CLIPtokenizer = tokenizer
        self.CLIPtext_encoder = text_encoder
        if not device == None:
            self.to(device)
        self.dtype = dtype
        self.to(dtype=dtype)

    def text_enc(self, prompts, maxlen=None):
        '''
        A function to take a textual promt and convert it into embeddings
        '''
        if maxlen is None: maxlen = self.CLIPtokenizer.model_max_length
        inp = self.CLIPtokenizer(prompts, padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt").to(self.device)
        return self.CLIPtext_encoder(inp.input_ids)[0].to(self.dtype)

    def forward(self, args_dict):
        '''
        g is guidance factor for diffusion model
        We likely won't change it, but leave it as an argument for now
        Maybe consider adding a dimension argument as well?
        '''
        if "train" in args_dict and args_dict["train"]:
            emb=args_dict["input_data_batch"].to(self.dtype)
            emb=emb.reshape(emb.shape[0], 1, 768)
#             g=args_dict["g"] if "g" in args_dict else 7.5
            timesteps=args_dict["timesteps"]
            noisy_latents=args_dict["noisy_latents"].to(self.dtype)
            res=self.unet(noisy_latents, timesteps, emb).sample
            return res
        else:
            emb=args_dict["input_data_batch"].to(self.dtype)
            dim=args_dict["dim"] if "dim" in args_dict else 512
            steps=30
            g=args_dict["g"] if "g" in args_dict else 7.5
#             print(emb.shape)
            bsz=emb.shape[0]
            emb = emb * torch.ones((bsz, 77, 768)).to(self.device,dtype=self.dtype)
            uncond = self.text_enc([""] * bsz, emb.shape[1])
#             print(uncond.shape)
            emb = torch.cat([uncond, emb])
            self.scheduler.set_timesteps(steps)
            latents = torch.randn((bsz, self.unet.in_channels, dim//8, dim//8))
            latents = latents.to(self.device, dtype=torch.float32) * self.scheduler.init_noise_sigma.to(self.device)
            
            with torch.no_grad():
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