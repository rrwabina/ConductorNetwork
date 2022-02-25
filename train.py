import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import torch.nn.utils as utils
import torchvision.transforms as transforms 
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import math
import matplotlib.pyplot as plt 

from segmentation_models.condnet.model import CondNet
from data_loader import load_data

torch.manual_seed(0)

def mpl_image_grid(images):
    n = min(images.shape[0], 16) 
    rows = 4
    cols = (n // 4) + (1 if (n % 4) != 0 else 0)
    figure = plt.figure(figsize=(2 * rows, 2 * cols))
    plt.subplots_adjust(0, 0, 1, 1, 0.001, 0.001)
    for i in range(n):
        plt.subplot(cols, rows, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if images.shape[1] == 3:
            vol = images[i].detach().numpy()
            img = [[[(1-vol[0,x,y]) * vol[1, x, y], (1 - vol[0, x, y]) * vol[2, x, y], 0] \
                            for y in range(vol.shape[2])] \
                            for x in range(vol.shape[1])]
            plt.imshow(img)
        else: 
            plt.imshow((images[i, 0] * 255).int(), cmap = 'gray')
    return figure

def prediction_tensorboard(prediction):
    image = prediction.permute(2, 3, 0, 1).detach().cpu().numpy()
    figure = plt.imshow(np.abs(image[:, :, 0, 0]), cmap = 'inferno')
    return figure

def log_to_tensorboard(writer, loss, slice_image, target, prediction, counter):
    writer.add_scalar('Loss', loss, counter)
    writer.add_figure('MRI T1W data', prediction_tensorboard(slice_image), global_step = counter)
    writer.add_figure('Predicted data', prediction_tensorboard(prediction), global_step = counter)
    writer.add_figure('Image data', mpl_image_grid(slice_image.float().cpu()), global_step = counter)
    writer.add_figure("Mask", mpl_image_grid(target.float().cpu()), global_step = counter)
    writer.add_figure('Prediction', 
                        mpl_image_grid(torch.argmax(prediction.cpu(), dim = 1, keepdim=True)), 
                        global_step=counter)

def train(epochs, inputs, targets):
    transform = transforms.Compose([
                        transforms.Normalize(
                                mean = [0.485],
                                std  = [0.229])
                                ])
    inputs, targets = transform(inputs), transform(targets)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CondNet()
    model.to('cpu')
    loss_function = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    predictions, losses = [], []

    tensorboard_train_writer = SummaryWriter('runs/condnet15')
    for epoch in range(0, epochs):
        running_loss = 0
        for idx, slice in enumerate(inputs[0:176]):
            slice_image = transform(slice.unsqueeze(0))
            prediction = model(slice_image)
            print(prediction.shape, targets.shape)
            prediction = prediction[:, 0:1, :, :]
            optimizer.zero_grad()
            loss = loss_function(prediction, targets)
            loss.backward()
            losses.append(loss)
            utils.clip_grad_norm_(model.parameters(), max_norm = 2.0, norm_type = 2)
            optimizer.step()
            scheduler.step(loss.item())
            running_loss += loss.item()
            if math.isnan(loss): 
                raise StopIteration
            counter = 100 * epoch + 100 * (idx/len(inputs))
            tensorboard_train_writer.add_scalar('CrossEntropy Loss', loss.item())
            predictions.append(prediction) 
            print(f"Epoch: {epoch} | Slice {idx} | Counter {counter:.2f} | Train loss: {loss.item():.7f} | Running loss: {running_loss:.4f}")
        print(f'Slice {idx} | Running loss: {running_loss:.4f}')
    print("\nTraining complete")
    tensorboard_train_writer.close()
    return predictions