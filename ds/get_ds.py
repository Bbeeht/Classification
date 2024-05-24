import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from types import SimpleNamespace
import os

def get_data(args):
    # Transform để chuyển đổi dữ liệu thành tensor
    transform = transforms.Compose([
        transforms.Resize((args.h, args.h)),  # Resize ảnh theo chiều cao h
        transforms.ToTensor()
    ])
    
    # Kiểm tra xem dataset đã tồn tại hay chưa, nếu chưa thì tải xuống
    dataset_path = os.path.join(args.root, 'cifar-10-batches-py')
    download = not os.path.exists(dataset_path)
    
    train_data = datasets.CIFAR10(
        root=args.root,
        train=True,
        download=True,  # Set to False to avoid re-downloading
        transform=transform
    )
    
    test_data = datasets.CIFAR10(
        root=args.root,
        train=False,
        download=True,  # Set to False to avoid re-downloading
        transform=transform
    )

    # Chia dataset huấn luyện thành tập huấn luyện và tập kiểm định
    val_size = int(len(train_data) * args.val_split)
    train_size = len(train_data) - val_size
    train_data, val_data = random_split(train_data, [train_size, val_size])
    
    # Cập nhật số lượng lớp trong args
    args.num_classes = 10
    
    return (train_data, val_data, test_data, args)

# Tạo một đối tượng args với các tham số cần thiết
args = {
    "root": "data",
    "val_split": 0.1,
    "h": 64, 
    "num_workers": 4
}
args = SimpleNamespace(**args)

# Gọi hàm để lấy dataset
train_ds, valid_ds, test_ds, args = get_data(args)

# In ra các kết quả để kiểm tra
print(f'Training dataset size: {len(train_ds)}')
print(f'Validation dataset size: {len(valid_ds)}')
print(f'Test dataset size: {len(test_ds)}')
print(f'Number of classes: {args.num_classes}')