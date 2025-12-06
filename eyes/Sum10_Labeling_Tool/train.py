import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np

# === 配置 ===
DATASET_DIR = 'dataset'
MODEL_PATH = 'sum10_model.pth'
IMG_SIZE = 64
BATCH_SIZE = 8
EPOCHS = 100     # 还是100轮，确保收敛
LEARNING_RATE = 0.001

# === 模型定义 (保持不变) ===
class SimpleDigitNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleDigitNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# === 数据加载 ===
class DigitDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        label = self.labels[idx]
        if self.transform: image = self.transform(image)
        return image, label

def load_data():
    image_paths = []
    labels = []
    # 强制重新扫描，确保读取所有数据
    classes = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    class_to_idx = {c: int(c) for c in classes if c.isdigit()}
    
    print(f"标签映射: {class_to_idx}")
    
    for class_name, class_idx in class_to_idx.items():
        folder_path = os.path.join(DATASET_DIR, class_name)
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(('.png', '.jpg')):
                image_paths.append(os.path.join(folder_path, fname))
                labels.append(class_idx)
    return image_paths, labels

# === 训练流程 ===
def train():
    img_paths, labels = load_data()
    if not img_paths: return

    # 划分数据集
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        img_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)), #稍微加大一点难度
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(DigitDataset(train_paths, train_labels, train_transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(DigitDataset(val_paths, val_labels, val_transform), batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    
    model = SimpleDigitNet(num_classes=10).to(device)
    
    # ==========================================
    # 🔥 核心修改：类别权重 (Class Weights)
    # ==========================================
    # 给 4 和 8 更高的权重，逼迫模型区分它们
    class_weights = torch.ones(10).to(device)
    class_weights[4] = 3.0  # 认错 4 的惩罚是平时的3倍
    class_weights[8] = 3.0  # 认错 8 的惩罚是平时的3倍
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)

    best_acc = 0.0
    print("\n🔥 开始带权重的强化训练...")

    for epoch in range(EPOCHS):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        scheduler.step(acc)

        if acc >= best_acc: # 只有更好或持平时才保存
            best_acc = acc
            torch.save(model.state_dict(), MODEL_PATH)
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Val Acc: {acc:.2f}% (Best: {best_acc:.2f}%)")

    print(f"✅ 训练结束。模型已保存。最佳准确率: {best_acc}%")

if __name__ == '__main__':
    train()