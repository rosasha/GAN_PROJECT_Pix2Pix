#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
# from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
# import config

import argparse
from Dataset import MapDataset
from gen_discr import Generator, Discriminator

from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

#torch.backends.cudnn.benchmark = True


# In[2]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# In[3]:


# Сохранение примеров работы GAN
def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(device), y.to(device)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()


# In[4]:


#parser = argparse.ArgumentParser()
#parser.add_argument('--lamb', type=float, default=100, help='lambda for L1 loss')
#parser.add_argument('--l_r', type=float, default=2e-4, help='lambda for L1 loss')

#params = parser.parse_args()


# In[5]:


lamb = 100
l_r = 2e-4
num_epochs = 250
batch_size = 1


# In[6]:


def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce,):
    
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(device)
        y = y.to(device)

        # Train Discriminator
        y_fake = gen(x)
        D_real = disc(x, y)
        D_real_loss = bce(D_real, torch.ones_like(D_real))
        D_fake = disc(x, y_fake.detach())
        D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss) / 2

        opt_disc.zero_grad()
        D_loss.backward()
        opt_disc.step()

        # Train generator
        D_fake = disc(x, y_fake)
        G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
        L1 = l1_loss(y_fake, y) * lamb # params.lamb
        G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )


# In[7]:


def main():
    
    disc = Discriminator(in_channels=3).to(device)
    gen = Generator(in_channels=3, features=64).to(device)
    
    disc.normal_weight_init(mean=0.0, std=0.02)
    gen.normal_weight_init(mean=0.0, std=0.02)
    
    opt_disc = optim.Adam(disc.parameters(), lr=l_r, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=l_r, betas=(0.5, 0.999))
    
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    train_dataset = MapDataset(root_dir='maps/train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
    )
    
    val_dataset = MapDataset(root_dir='maps/val')
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    for epoch in range(num_epochs):
        train_fn(disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE,)

        if epoch % 10 == 0:
            save_some_examples(gen, val_loader, epoch, folder="./")
    save_some_examples(gen, val_loader, epoch, folder="./")


# In[ ]:


if __name__ == "__main__":
    main()


# In[ ]:


gen_model =  Generator().state_dict()
PATH = "state_dict_gen_model_rem.pt" # Сохраним наш обученный на 200 эпохах генератор   
torch.save(gen_model, PATH)


# In[ ]:


# восстанавливаем генератор
PATH = "state_dict_gen__model_rem.pt" 
gen = Generator().to(device)
gen.load_state_dict(torch.load(PATH))


# In[ ]:





# In[ ]:




