import sys 
import os 
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import kans
# Train on MNIST
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import time 
import numpy as np
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import argparse

parser = argparse.ArgumentParser()
# 'KAN' or 'MLP'
parser.add_argument('--model', type=str, default='MLP')
parser.add_argument('--epoch', type=int, default=10)
args = parser.parse_args()

# Load MNIST
transform   = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
trainset    = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
valset      = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
trainloader = DataLoader(trainset, batch_size=256, shuffle=True)
valloader   = DataLoader(valset, batch_size=256, shuffle=False)


    
# Define model
model   = getattr(kans,args.model)([28 * 28, 128, 10])

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Define optimizer
if args.model =='MNISTFourierKAN':
    optimizer = optim.LBFGS(model.parameters(), lr=1e-2)
else:
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# Define learning rate scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

# Define loss
criterion = nn.CrossEntropyLoss()

time_list = []
for epoch in range(args.epoch):
    # Train
    model.train()
    sta_time = time.time()
    with tqdm(trainloader) as pbar:
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.view(-1, 28 * 28).to(device), labels.to(device)

            if args.model == 'MNISTFourierKAN':
                def closure():
                    optimizer.zero_grad()
                    output  = model(images)
                    loss    = criterion(output, labels)
                    loss.backward()
                    return loss
                optimizer.step(closure)
                loss        = closure()
            else:
                optimizer.zero_grad()
                output  = model(images)
                loss    = criterion(output, labels.to(device))
                loss.backward()
                optimizer.step()

            pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
    end_time = time.time()
    time_list.append(end_time-sta_time)

    # Validation
    model.eval()
    val_loss        = 0
    val_accuracy    = 0
    with torch.no_grad():
        for images, labels in valloader:
            images          = images.view(-1, 28 * 28).to(device)
            output          = model(images)
            val_loss        += criterion(output, labels.to(device)).item()
            val_accuracy    += (
                (output.argmax(dim=1) == labels.to(device)).float().mean().item()
            )
    val_loss        /= len(valloader)
    val_accuracy    /= len(valloader)

    # Update learning rate
    scheduler.step()

    print(
        f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}, avg time:{np.mean(time_list)} s"
    )


# toy testing
model       = getattr(kans,args.model)([28 * 28, 64, 10])
test_x      = valset[0][0].view(-1, 28 * 28)
inf_time    = []
for i in range(500):
    inf_sta_time    =  time.time()
    res = model(test_x)
    inf_end_time    =   time.time()
    inf_time.append(inf_end_time-inf_sta_time)

def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    return (total / 1e6)

print(f'{args.model} | Averaged Inference Time:{np.mean(inf_time)}')
flops = FlopCountAnalysis(model, test_x)
print(f"{args.model} | MACs: %.4f M " % (flops.total()/ 1e6))
print(f"{args.model} | Params: %.4f M" % print_model_parm_nums(model))
print(parameter_count_table(model))