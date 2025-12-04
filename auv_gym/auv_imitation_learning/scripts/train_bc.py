#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os
from auv_imitation_learning.models import MLPPolicy


class AUVDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)  # expert data
        self.inputs = torch.FloatTensor(data["inputs"])
        self.labels = torch.FloatTensor(data["labels"])  # output (cmd_vel)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # PyTorch calls this repeatedly to fetch data.
        return self.inputs[idx], self.labels[idx]


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = AUVDataset(args.dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    input_dim = dataset.inputs.shape[1]
    output_dim = dataset.labels.shape[1]

    # create the MLP
    model = MLPPolicy(input_dim, output_dim, hidden_dim=args.hidden_dim).to(
        device
    )  # move to GPU if available
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(
        f"Model Architecture: Input={input_dim}, Hidden={args.hidden_dim}, Output={output_dim}"
    )
    print(f"Training on {len(dataset)} samples for {args.epochs} epochs...")

    # Training Loop
    for epoch in range(args.epochs):
        total_loss = 0
        # batch iteration
        # for example, inputs is of shape (batch_size, input_dim)
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.6f}")

    # Save Model
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    model.save(args.output_path)
    print(f"Model saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BC Agent")
    parser.add_argument("--dataset", required=True, help="Path to NPZ dataset")
    parser.add_argument(
        "--output_path",
        default="models/bc_policy.pth",
        help="Path to save trained model",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)

    args = parser.parse_args()

    train(args)
