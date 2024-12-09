import torch
import torch.nn.functional as F

import math
import numpy as np

from PIL import Image

def forward_diffusion(x_0, t, beta_schedule, s = 8e-3, function = "linear", device = 'cpu'):
   
    x_0 = x_0.to(device)
    beta_schedule = beta_schedule.to(device)

    with torch.no_grad():

        if function == "linear":
            beta_t = beta_schedule[t].to(device)
        else:
            f_t = (torch.arange(len(beta_schedule), device=device).float() / (1 + s)) * (math.pi / 2)
            f_t = torch.cos(f_t).pow(2)
            a_h = f_t / f_t[0]

            beta_t = a_h[1:] / a_h[:-1]
            beta_t = 1 - torch.clamp(beta_t, max = 0.999)
            beta_t = torch.cat([beta_t, beta_t[-1].unsqueeze(0)], dim=0) if len(beta_t) < len(a_h) else beta_t

        alpha_t = 1 - beta_t
        alpha_cumprod_t = torch.cumprod(alpha_t, dim=0)
        alpha_cumprod_t = alpha_cumprod_t.view(-1, 1, 1, 1)
        #alpha_cumprod_t = alpha_cumprod_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(torch.tensor(x_0))

        random_noise = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_cumprod_t) * x_0 + (torch.sqrt(1 - alpha_cumprod_t) + 1e-10) * random_noise

    return x_t, random_noise


def mse_loss(model, x_0, t):
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  x_0 = x_0.to(device)
  t = t.to(device)

  x_t, eps = forward_diffusion(x_0, t, beta_schedule=torch.linspace(0.0, 1.0, len(t)), function = "cosine", device=device)
  eps_h = model(x_t.float(), t.float())
  return F.mse_loss(eps, eps_h)

def resize_images(data, new_size=(32, 32)):
  """
  Resize image
  """
  labels = data.iloc[:, 0].values
  images = data.iloc[:, 1:].values / 255.0

  resized_images = []
  for image in images:
      image = image.reshape(28, 28)
      image = Image.fromarray(image.astype(np.uint8))
      image = image.resize(new_size, Image.LANCZOS)
      resized_images.append(np.array(image))

  return np.array(resized_images), labels

