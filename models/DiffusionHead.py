import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionHead(nn.Module):
	def __init__(self, scheduler, unet):
		super(EEGEncoder, self).__init__()
		self.scheduler = scheduler
		self.unet = unet

	def forward(self, embd, steps):
		uncond = text_enc_tuned([""] * bs, emb.shape[1])
		emb = torch.cat([uncond, emb])
		self.scheduler.set_timesteps(steps)
		latents = latents.to(device).half() * self.scheduler.init_noise_sigma

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