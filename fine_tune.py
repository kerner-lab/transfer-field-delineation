import torch
import torch.nn as nn
import numpy as np
import os
from models.unet_models import UNetResNet50_9, UNetResNet50_3
from torch.utils.tensorboard import SummaryWriter
from utils.dataloader import ThreeSeasonDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import argparse

parser = argparse.ArgumentParser(description='Fine-Tune script for the UNetResNet50_X model.')
parser.add_argument('--config', '-c', type=str, default='configs/config.yaml', help='Path to the config file.')

args = parser.parse_args()

# Load config file
with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


log_dir = f'logs'
writer = SummaryWriter(log_dir)

# Hyperparameters
# Define Hyperparameters
num_epochs = config['fine_tuning']['num_epochs']
batch_size = config['fine_tuning']['batch_size']
learning_rate = config['fine_tuning']['learning_rate']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model
model_name = config['model']['name']
save_name = config['model']['save_name']
model = globals()[model_name]().load_state_dict(torch.load(f'logs/{save_name}.pt'))
model.to(device)

# Define loss function
criterion = nn.BCEWithLogitsLoss()

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Load data
dataset = globals()[config['data']['dataset']](root_dir=config['data']['fine_tuning_path'])

# split the dataset into train and test and validation
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# split the train dataset into train and validation
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# Create dataloader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Train model
total_step = len(train_dataloader)

for epoch in tqdm(range(num_epochs)):
    model.train()
    epoch_loss = 0
    for i_batch, sample_batched in enumerate(train_dataloader):
        # Get batch
        images = sample_batched['image'].to(device)
        filled = sample_batched['filled'].to(device)
        border = sample_batched['border'].to(device)

        # Forward pass
        filled_pred, border_pred = model(images)
        filled_loss = criterion(filled_pred, filled)
        border_loss = criterion(border_pred, border)
        loss = filled_loss + border_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss
        if (i_batch + 1) % 10 == 0:
            tqdm.write('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i_batch + 1, total_step, loss.item()))

        # Log loss
        writer.add_scalar('Loss/train', loss.item(), epoch * total_step + i_batch)

    # Evaluate model
    model.eval()
    with torch.no_grad():
        total_val_loss = 0
        for i_batch, sample_batched in enumerate(val_dataloader):
            # Get batch
            images = sample_batched['image'].to(device)
            filled = sample_batched['filled'].to(device)
            border = sample_batched['border'].to(device)

            # Forward pass
            filled_pred, border_pred = model(images)
            filled_loss = criterion(filled_pred, filled)
            border_loss = criterion(border_pred, border)
            loss = filled_loss + border_loss
            total_val_loss += loss.item()

        print('Epoch [{}/{}], Val Loss: {:.4f}'.format(epoch + 1, num_epochs, total_val_loss / len(val_dataloader)))
        # Log loss
        writer.add_scalar('Loss/val', total_val_loss / len(val_dataloader), epoch)
        

    # Save the best model
    if epoch == 0:
        best_loss = total_val_loss
        torch.save(model.state_dict(), f'logs/{save_name}.pt')
    else:
        if total_val_loss < best_loss:
            best_loss = total_val_loss
            torch.save(model.state_dict(), f'logs/{save_name}.pt')

# Test model
model.eval()
with torch.no_grad():
    total_test_loss = 0
    for i_batch, sample_batched in enumerate(test_dataloader):
        # Get batch
        images = sample_batched['image'].to(device)
        filled = sample_batched['filled'].to(device)
        border = sample_batched['border'].to(device)

        # Forward pass
        filled_pred, border_pred = model(images)
        filled_loss = criterion(filled_pred, filled)
        border_loss = criterion(border_pred, border)
        loss = filled_loss + border_loss
        total_test_loss += loss.item()

    # Log loss
    writer.add_scalar('Loss/test', total_test_loss / len(test_dataloader), epoch)

# Close Tensorboard writer
writer.close()    