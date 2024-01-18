"""Inference script for the UNetResNet50_X model."""
from zmq import device
from models.unet_models import UNetResNet50_9, UNetResNet50_3
import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from utils.metrics import calculate_metrics

parser = argparse.ArgumentParser(description='Inference script for the UNetResNet50_X model.')
parser.add_argument('--model', type=str, default='3', help='Number of season\'s images to use (1 or 3)')
parser.add_argument('--model_path', type=str, default='logs/UNetResNet50_9.pt', help='Path to model weights')
parser.add_argument('--input_path', type=str, default='test_data/', help='Path to input images')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
if args.model == '3':
    model = UNetResNet50_9()
elif args.model == '1':
    model = UNetResNet50_3()
else:
    raise ValueError('Model must be 1 or 3')

try:
    model.load_state_dict(torch.load(args.model_path, map_location=device))
except FileNotFoundError:
    print('Model not found. Please check the path to the model weights.')
    exit(1)

# Load input images
images = np.array([])
for filename in os.listdir(args.input_path):
    if filename.endswith('.jpeg'):
        img = Image.open(os.path.join(args.input_path, filename))
        img = np.array(img)
        img = np.transpose(img, (2, 0, 1))
        if images.size == 0:
            images = img
        else:
            images = np.concatenate((images, img), axis=0)

# Convert to tensor
input_tensor = torch.from_numpy(images).float().unsqueeze(0)

# Perform inference
model.eval()
with torch.no_grad():
    output1, output2 = model(input_tensor)
    output1 = output1.squeeze(0).squeeze(0).numpy()
    output2 = output2.squeeze(0).squeeze(0).numpy()

# Plot results
# plt.imshow(output)
# plt.show()


# Save results
plt.imsave('test_data/output1.png', output1, cmap='gray')
plt.imsave('test_data/output2.png', output2, cmap='gray')

# Load ground truth
gt1 = Image.open('test_data/masks.png')
gt1 = np.array(gt1)
gt1 = np.expand_dims(gt1, axis=0)
gt1[gt1 > 0] = 1
gt1 = gt1.astype(np.float32)

gt2 = Image.open('test_data/masks_filled.png')
gt2 = np.array(gt2)
gt2 = np.expand_dims(gt2, axis=0)
gt2[gt2 > 0] = 1
gt2 = gt2.astype(np.float32)

# Calculate metrics
f1_score1, accuracy1, iou1, precision1 = calculate_metrics(torch.from_numpy(output1).unsqueeze(0), torch.from_numpy(gt1))
f1_score2, accuracy2, iou2, precision2 = calculate_metrics(torch.from_numpy(output2).unsqueeze(0), torch.from_numpy(gt2))

print(f'Boundary    : F1 Score: {f1_score1}, Accuracy: {accuracy1}, IoU: {iou1}, Precision: {precision1}')
print(f'Filled      : F1 Score: {f1_score2}, Accuracy: {accuracy2}, IoU: {iou2}, Precision: {precision2}')