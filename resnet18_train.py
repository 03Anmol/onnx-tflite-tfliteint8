import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = '/home/anmol/Documents/waste/Dataset-Airport'
dataset = datasets.ImageFolder(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(dataset.classes))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

best_loss = float('inf')
os.makedirs("/home/anmol/Documents/waste/model_weights", exist_ok=True)

epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    total = len(dataloader)
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        percent = (batch_idx + 1) / total * 100
        print(f"Epoch {epoch+1}/{epochs} - Batch {batch_idx+1}/{total} ({percent:.2f}%) - Loss: {loss.item():.4f}")
    epoch_loss = running_loss / total
    print(f"Epoch {epoch+1} completed with avg loss: {epoch_loss:.4f}")
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), "/home/anmol/Documents/waste/model_weights/resnet18_best.pth")
