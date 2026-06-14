import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

hub_expr = pd.read_csv(
    "/home/achyuta_shiva_sai/ACC-Biomarker-Discovery/results/hub_gene_matrix.csv",
    index_col=0
)

X = hub_expr.T.values.astype(np.float32)
sample_names = hub_expr.columns.tolist()
print(f"Input matrix: {X.shape[0]} samples x {X.shape[1]} genes")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

input_dim  = X.shape[1]
latent_dim = 16

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

model = Autoencoder(input_dim, latent_dim).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
epochs = 500
losses = []

model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    x_recon, z = model(X_tensor)
    loss = criterion(x_recon, X_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs} — Loss: {loss.item():.6f}")

print("Training complete")

model.eval()
with torch.no_grad():
    _, latent = model(X_tensor)
    latent_np = latent.cpu().numpy()

print(f"Latent matrix: {latent_np.shape}")

latent_df = pd.DataFrame(
    latent_np,
    index=sample_names,
    columns=[f"latent_{i+1}" for i in range(latent_dim)]
)
latent_df.to_csv(
    "/home/achyuta_shiva_sai/ACC-Biomarker-Discovery/results/autoencoder_latent.csv"
)

pd.DataFrame({"epoch": range(1, epochs+1), "loss": losses}).to_csv(
    "/home/achyuta_shiva_sai/ACC-Biomarker-Discovery/results/autoencoder_loss.csv",
    index=False
)

print("Outputs saved")
