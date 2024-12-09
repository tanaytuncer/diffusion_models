import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from Unet import UNet
from utilities import mse_loss

from torch.utils.data import Subset
from tqdm import tqdm



# Setting up parameters

n = 2000
n_batch = 64
epochs = 10
lr = 5e-5 #1e-4
T = 20

print(f"Number of processed images: {n}, Batch Size: {n_batch}, Epochs: {epochs}, Step size {T}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

# Transform images 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

def main():

    #Load training data
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    subset = Subset(train_data, range(n))

    X_train_loader = DataLoader(subset, batch_size=n_batch, shuffle=True, drop_last=True)

    data = iter(X_train_loader)
    images, _ = next(data)    

    # Init model
    model = UNet(img_dim = 3).to(device=device)
    model.train()

    optimizer = Adam(model.parameters(), lr=lr)

    # Training model
    for i in tqdm(range(100)):
        for epoch in range(epochs):
            for step, batch in enumerate(images.to(device=device)):
                optimizer.zero_grad()
                t = torch.randint(0, T, (n_batch,)).long()
                loss = mse_loss(model, batch, t)
                loss.backward()
                optimizer.step()
                if step % 10 == 0:
                    print(f'Epoch: {epoch}, Batch step: {step}, Loss: {loss.item()}')

    # Save trained model
    model_path = '/home/tanay.tuncer/diffusion_model/models/240124_model_2_res.pth'
    torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    main()