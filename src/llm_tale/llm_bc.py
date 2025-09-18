import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BC, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BCDataset(Dataset):
    def __init__(self, observations, actions):
        self.observations = observations
        self.actions = actions

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


# Training loop
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for obs, acts in loader:
        obs = obs.to(device)
        acts = acts.to(device)
        optimizer.zero_grad()
        outputs = model(obs)
        loss = criterion(outputs, acts)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# Validation loop
def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for obs, acts in loader:
            obs = obs.to(device)
            acts = acts.to(device)
            outputs = model(obs)
            loss = criterion(outputs, acts)
            total_loss += loss.item()
    return total_loss / len(loader)
