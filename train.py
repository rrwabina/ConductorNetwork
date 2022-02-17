import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import torchvision.transforms as transforms 
from torch.utils.tensorboard import SummaryWriter

from CondNet import ConductorNetwork
from utils import image_size, image_loader, view_image
from data_loader import load_split_data

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

def log_to_tensorboard(writer, loss, data, target, prediction, counter):
    writer.add_scalar('Loss', loss, counter)
    writer.add_figure('Image data', 
                        mpl_image_grid(data.float().cpu()), global_step = counter)
    writer.add_figure("Mask", 
                        mpl_image_grid(target.float().cpu()), global_step = counter)
    writer.add_figure('Prediction', 
                        mpl_image_grid(torch.argmax(prediction.cpu(), dim = 1, keepdim=True)), 
                        global_step=counter)


def train(epochs):
    train_loader = load_split_data('train')
    transform = transforms.Compose([
                        transforms.Normalize(
                                mean=[0.485],
                                std =[0.229]),
                        transforms.Resize(48) ])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CondNet()
    model.to(device)

    loss_function = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    predictions, losses = [], []
    for epoch in range(0, epochs):
        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            t1w, t2w = image_loader(train_loader, idx)
            t1w, t2w = image_size(t1w, 1), image_size(t2w, 2)
            t1w, t2w = transform(t1w), transform(t2w)

            data = t1w.to(device, dtype = torch.cuda.FloatTensor)
            target = batch['seg'].to(device, dtype = torch.cuda.LongTensor)

            prediction = model(data, data)
            predictions.append(prediction)

            loss = loss_function(prediction, target)
            loss.backward()
            losses.append(loss)

            optimizer.step()

            tensorboard_train_writer = SummaryWriter(comment = '_train')

            if (i % 5) == 0:
                print(f"Epoch: {epoch} | Batch {i} | Train loss: {loss:.5f}")
                counter = 100 * epoch + 100 * (i/len(train_loader))
                log_to_tensorboard(
                    tensorboard_train_writer,
                    loss,
                    data,
                    target,
                    prediction,
                    counter)
    print("\nTraining complete")
