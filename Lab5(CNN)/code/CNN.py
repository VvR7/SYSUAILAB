import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
import os
import random
import shutil
import matplotlib.pyplot as plt
def split_train_val(train_dir, val_dir, split_ratio=0.15, seed=42):
    random.seed(seed)
    os.makedirs(val_dir, exist_ok=True)

    classes = os.listdir(train_dir)
    for cls in classes:
        cls_train_dir = os.path.join(train_dir, cls)
        cls_val_dir = os.path.join(val_dir, cls)

        if not os.path.isdir(cls_train_dir):
            continue  
        os.makedirs(cls_val_dir, exist_ok=True)

        images = [f for f in os.listdir(cls_train_dir) if f.endswith('.jpg')]

        # 随机选择15%的图片
        num_val = max(1, int(len(images) * split_ratio)) 
        val_images = random.sample(images, num_val)

        for img_name in val_images:
            src_path = os.path.join(cls_train_dir, img_name)
            dst_path = os.path.join(cls_val_dir, img_name)

            shutil.move(src_path, dst_path)  

        print(f"类别 {cls}：共{len(images)}张，移动{num_val}张到验证集。")
# 训练集用的数据增强（增加模型泛化能力）
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪并缩放
    transforms.RandomHorizontalFlip(p=0.5),  # 50%概率水平翻转
    transforms.RandomRotation(15),  # 随机旋转±15度
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
# train_transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # 直接Resize到224
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#     ),
# ])
# 验证/测试集的预处理（保持固定，不做随机变换）
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 直接Resize到224
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

script_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(script_dir, "image", "train")
val_dir = os.path.join(script_dir, "image", "val")

train_dataset=datasets.ImageFolder(
    root=train_dir,
    transform=train_transform
)

val_dataset=datasets.ImageFolder(
    root=val_dir,
    transform=val_transform
)

train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=1,shuffle=False)

class CNN(nn.Module):
    def __init__(self, num_classes=5):  # 5 类
        super(CNN, self).__init__()
        
        # 第一个卷积块
        self.conv1 = nn.Conv2d(3,64,5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 尺寸减半
        
        # 第二个卷积块
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # 第三个卷积块
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 28 * 28, 256)  # 展平后到 512维
        self.fc2 = nn.Linear(256, 128)    
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)
    def forward(self, x):
        # 第一块
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # [batch, 10, 112, 112]
        
        # 第二块
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # [batch, 20, 56, 56]
        
        # 第三块
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # [batch, 40, 28, 28]
        
        # 展平
        x = x.view(x.size(0), -1)  # Flatten
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x

model = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # 将模型移动到正确的设备上
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)  # 稍微提高初始学习率
# 添加学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',     
    factor=0.5,       
    patience=5,       
    verbose=True,        
    min_lr=1e-6         
)

def train(epochs):
    train_history_loss=[]
    val_history_loss=[]
    best_acc=0
    stop_count=0
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            model.train()
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
        print(f"Epoch {epoch} Loss: {epoch_loss/len(train_loader):.4f}")
        
        # 评估模型并更新学习率
        acc, loss = evaluate()
        scheduler.step(acc)  # 根据验证集准确率调整学习率
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_model.pth')
            stop_count = 0
        else:
            stop_count += 1
        val_history_loss.append(loss)
        train_history_loss.append(epoch_loss/len(train_loader))
        
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.2e}")
        
        if stop_count >= 50:
            print(f"Early stopping at epoch {epoch}")
            break
    return train_history_loss,val_history_loss
def evaluate():
    model.eval()
    with torch.no_grad():
        correct = 0
        loss=0
        total = 0
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            loss+=criterion(output,target).item()
    print(f"Accuracy of the model on the {total} val images: {100 * correct / total:.2f}%")
    return 100 * correct / total,loss/total
def infer(img_path,idx_to_class):
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    with torch.no_grad():
        img = Image.open(img_path)
        img = val_transform(img).unsqueeze(0)
        img = img.to(device)
        output = model(img)
        _, predicted = torch.max(output.data, 1)
        return idx_to_class[predicted.item()]
def test(idx_to_class):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(script_dir, "image", "test")
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            file_path=os.path.join(root,file)
            predicted_class=infer(file_path,idx_to_class)
            print(f'file:{file} predict_class:{predicted_class}')
    
    
if __name__ == "__main__":
    train_history_loss,val_history_loss=train(epochs=500)
    evaluate()
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    
    test(idx_to_class)
    


    ######可视化
    plt.figure(figsize=(12, 4))
    
    # 训练损失子图
    plt.subplot(1, 2, 1)
    plt.plot(train_history_loss, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.legend()
    
    # 验证损失子图 
    plt.subplot(1, 2, 2)
    plt.plot(val_history_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss') 
    plt.title('Validation Loss History')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
