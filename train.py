import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from conf import settings
from utils.utils import WarmUpLR
from utils.model_utils import get_network
from utils.data_utils import get_val_dataloader, get_train_dataloader
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(epoch):

    start = time.time()
    net.train()
    for batch_index, (images, fileid, labels) in enumerate(ocean_training_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()

        labels = labels.to(device)
        images = images.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(ocean_training_loader) + batch_index + 1

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.bs + len(images),
            total_samples=len(ocean_training_loader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)


    finish = time.time()
    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch):

    start = time.time()
    net.eval()

    val_loss = 0.0 # cost function error
    correct = 0.0

    for (images, fileid, labels) in ocean_val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        loss = loss_function(outputs, labels)
        val_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()

    print('Evaluating Network.....')
    print('Val set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        val_loss / len(ocean_val_loader.dataset),
        correct.float() / len(ocean_val_loader.dataset),
        finish - start
    ))
    print()

    #add informations to tensorboard
    writer.add_scalar('Val/Average loss', val_loss / len(ocean_val_loader.dataset), epoch)
    writer.add_scalar('Val/Accuracy', correct.float() / len(ocean_val_loader.dataset), epoch)

    return correct.float() / len(ocean_val_loader.dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-root', type=str, default='/cache/', help='root path')
    parser.add_argument('-net', type=str, default='resnet18', help='net type')
    parser.add_argument('-bs', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()

    net = get_network(args, pretrained=True)

    train_csv = pd.read_csv(args.root+'training.csv')
    val_csv = pd.read_csv(args.root+'annotation.csv')
    #data preprocessing:
    ocean_training_loader = get_train_dataloader(
        img_path=args.root+'data/',
        train_csv=train_csv,
        num_workers=4,
        batch_size=args.bs,
        shuffle=True
    )

    ocean_val_loader = get_val_dataloader(
        img_path=args.root+'data/',
        val_csv=val_csv,
        num_workers=4,
        batch_size=args.bs,
        shuffle=False
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(ocean_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    for epoch in range(1, settings.EPOCH):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train(epoch)
        acc = eval_training(epoch)

        #start to save best performance model after epoch MILESTONES[1]
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

    writer.close()
