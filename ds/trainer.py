import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from alive_progress import alive_it
from get_ds import get_data
import json
import hashlib
import os
from types import SimpleNamespace

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = x.view(-1, 64 * 8 * 8)  
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x) 
        return x

def get_hash(args):
    args_str = json.dumps(vars(args), sort_keys=True)
    args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
    return args_hash

    model = SimpleCNN(num_classes=args.num_classes).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Params: {total_params}")
    total_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Params: {total_train_params}")

   
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, len(train_ld) * args.epoch)

    criterion = nn.CrossEntropyLoss()

    best_valid_acc = 0  

    for epoch in range(args.epoch):
        log_dict = {}  
        
        model.train() 
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for img, lbl in alive_it(train_ld):
            img = img.to(device)
            lbl = lbl.to(device)

            optimizer.zero_grad() 
            output = model(img)  
            loss = criterion(output, lbl) 
            loss.backward()  
            optimizer.step()  
            
            scheduler.step() 
            
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)  
            total_correct += (predicted == lbl).sum().item()  
            total_samples += lbl.size(0)  
        
        train_loss = total_loss / len(train_ld)  
        train_acc = total_correct / total_samples  
        
        log_dict['train/loss'] = train_loss  
        log_dict['train/accuracy'] = train_acc  

        print(f"Epoch: {epoch} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_acc:.4f}")

        model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for img, lbl in alive_it(valid_ld):
                img = img.to(device)
                lbl = lbl.to(device)

                output = model(img)
                loss = criterion(output, lbl)

                total_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total_correct += (predicted == lbl).sum().item()
                total_samples += lbl.size(0)
        
        valid_loss = total_loss / len(valid_ld)  
        valid_acc = total_correct / total_samples  

        log_dict['valid/loss'] = valid_loss  
        log_dict['valid/accuracy'] = valid_acc  

        print(f"Epoch: {epoch} - Valid Loss: {valid_loss:.4f} - Valid Accuracy: {valid_acc:.4f}")

        save_dict = {
            'args': vars(args),
            'model_state_dict': model.state_dict()
        }

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(save_dict, best_model_path)
        torch.save(save_dict, last_model_path)

        if args.log:
            wandb.log(log_dict)
    
    if args.log:
        run.log_artifact(wandb.Artifact(name=f'{run_name}-best-model', type='model', metadata={'epoch': args.epoch}), best_model_path)
        run.log_artifact(wandb.Artifact(name=f'{run_name}-last-model', type='model', metadata={'epoch': args.epoch}), last_model_path)
