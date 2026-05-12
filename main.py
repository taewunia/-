import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from torchmetrics import Accuracy
from tqdm import tqdm
from torch.utils.data import DataLoader


device = "cuda" if torch.device.cuda.is_available() else "cpu"
print(f"현재 사용중인 기기: {device}")

EPOCH = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
P = 0.3

transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def visualize(model=None, device=None, DL=None):
    model.eval()
    with torch.no_grad():
        sample, labels = next(iter(DL))
        img_tensor = sample[0].unsqueeze(0).to(device)
        label = labels[0].item()

        classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

        model_on_device = model.to(device)
        output = model_on_device.layer1(img_tensor)
        output = output.squeeze(0).cpu().numpy()

        fig = plt.figure(figsize=(12, 10))
        grid = plt.GridSpec(5, 4, hspace=0.4, wspace=0.3)

        ax_main = fig.add_subplot(grid[0, :])
        origin_img = sample[0].cpu().permute(1, 2, 0).numpy()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        origin_img = origin_img * std + mean
        ax_main.imshow(origin_img.clip(0, 1))
        ax_main.set_title(f"Original Image (Label: {classes[label]})", fontsize=14, fontweight='bold')
        ax_main.axis('off')

        for i in range(16):
            ax = fig.add_subplot(grid[1 + i // 4, i % 4])
            ax.imshow(output[i], cmap='viridis')
            ax.axis('off')
            ax.set_title(f"Filter {i + 1}", fontsize=10)

        plt.show()


train_stl_dataset = datasets.STL10('dataset', split='train', download=True, transform=transform)
test_stl_dataset = datasets.STL10('dataset', split='test', download=True, transform=transform)

train_DL = DataLoader(train_stl_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_DL = DataLoader(test_stl_dataset, batch_size=BATCH_SIZE, shuffle=False)


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_layer = nn.Sequential(
            nn.Dropout(P),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avg_pool(out)
        out_flatten = out.view(out.size(0), -1)
        output = self.fc_layer(out_flatten)
        return output


model = MyModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
accuracy_metric = Accuracy(task="multiclass", num_classes=10).to(device)

train_loss_history, test_loss_history = [], []
train_acc_history, test_acc_history = [], []

for epoch in range(EPOCH):
    model.train()
    train_bar = tqdm(train_DL, desc=f'Epoch {epoch + 1}/{EPOCH} [Train]', colour='green')
    total_train_loss = 0
    accuracy_metric.reset()

    for x, y in train_bar:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        accuracy_metric.update(output, y)
        train_bar.set_postfix(loss=loss.item())

    avg_train_loss = total_train_loss / len(train_DL)
    train_acc = accuracy_metric.compute().item()

    model.eval()
    test_bar = tqdm(test_DL, desc=f'Epoch {epoch + 1}/{EPOCH} [Test]', colour='red', leave=False)
    total_test_loss = 0
    accuracy_metric.reset()

    with torch.no_grad():
        for x, y in test_bar:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            total_test_loss += loss.item()
            accuracy_metric.update(output, y)
            test_bar.set_postfix(loss=loss.item())

    visualize(model=model, device=device, DL=test_DL)
    avg_test_loss = total_test_loss / len(test_DL)
    test_acc = accuracy_metric.compute().item()

    train_loss_history.append(avg_train_loss)
    test_loss_history.append(avg_test_loss)
    train_acc_history.append(train_acc)
    test_acc_history.append(test_acc)

    tqdm.write(
        f"➔ Result: Train Loss {avg_train_loss:.4f}, Acc {train_acc * 100:.2f}% | Test Loss {avg_test_loss:.4f}, Acc {test_acc * 100:.2f}%")

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label='Train Loss')
plt.plot(test_loss_history, label='Test Loss')
plt.title('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(train_acc_history, label='Train Acc')
plt.plot(test_acc_history, label='Test Acc')
plt.title('Accuracy')
plt.legend()
plt.show()