import math
import torch
import numpy as np
from utilities import forward_diffusion
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

def reverse_diffusion(model, n_batch, T, shape):

    x = torch.randn((3, 32, 32))
    t = torch.randint(0, T, (n_batch,)).long()

    beta_schedule = torch.linspace(0.0, 1.0, T)
    alpha_schedule = 1 - beta_schedule
    alpha_cumprod_schedule = torch.cumprod(alpha_schedule, dim=0)

    for i in reversed(range(T)):
        #x, _ = forward_diffusion(x, torch.tensor([i], beta_schedule))
        x, _ = forward_diffusion(x, t, beta_schedule)
        x = model(x, torch.tensor([i], dtype=torch.float32))
        if i > 0:
            noise = torch.randn_like(x)
            alpha_cumprod = alpha_cumprod_schedule[i-1]
            x = (1 / torch.sqrt(alpha_schedule[i])) * (x - (1-alpha_schedule[i]) / torch.sqrt(1-alpha_cumprod) * noise)

    return x

def resize_images(images, shape):
    images_resized = []
    for img in images:
        img_pil = transforms.ToPILImage()(img)
        img_resized = img_pil.resize(shape, Image.BILINEAR)
        img_tensor = transforms.ToTensor()(img_resized)
        images_resized.append(img_tensor)
    return images_resized

def transform_img(images):

    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299)
    ])

    transformed_images = []

    for img in images:
        img = to_pil_image(img)
        transformed_image = transform(img)
        transformed_image = to_tensor(transformed_image)
        transformed_images.append(transformed_image)

    transformed_images = torch.stack(transformed_images)
    #transformed_imgs = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(transformed_images)
    transformed_images = (transformed_images * 255).byte()

    return transformed_images

