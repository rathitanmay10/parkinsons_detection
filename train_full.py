import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import joblib
import os

# =======================
# 1. Load Dataset
# =======================
data = pd.read_csv("data/parkinsons.data")
X = data.drop(columns=["name", "status"])
y = data["status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =======================
# 2. GAN Definitions
# =======================
latent_dim = 32
n_features = X_train_scaled.shape[1]
n_classes = 2

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 64),
            nn.ReLU(),
            nn.Linear(64, n_features),
        )
    def forward(self, noise, labels):
        c = self.label_emb(labels)
        x = torch.cat([noise, c], dim=1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.model = nn.Sequential(
            nn.Linear(n_features + n_classes, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    def forward(self, features, labels):
        c = self.label_emb(labels)
        x = torch.cat([features, c], dim=1)
        return self.model(x)

generator = Generator()
discriminator = Discriminator()
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.001)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.001)

# =======================
# 3. Train GAN
# =======================
X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_train.values, dtype=torch.long)

epochs = 500
for epoch in range(epochs):
    # Real data
    idx = np.random.randint(0, X_tensor.shape[0], 32)
    real_samples = X_tensor[idx]
    real_labels = y_tensor[idx]
    real_validity = torch.ones(32, 1)

    # Fake data
    z = torch.randn(32, latent_dim)
    gen_labels = torch.randint(0, 2, (32,))
    fake_samples = generator(z, gen_labels)
    fake_validity = torch.zeros(32, 1)

    # Train Discriminator
    optimizer_D.zero_grad()
    real_loss = criterion(discriminator(real_samples, real_labels), real_validity)
    fake_loss = criterion(discriminator(fake_samples.detach(), gen_labels), fake_validity)
    d_loss = (real_loss + fake_loss) / 2
    d_loss.backward()
    optimizer_D.step()

    # Train Generator
    optimizer_G.zero_grad()
    validity = discriminator(fake_samples, gen_labels)
    g_loss = criterion(validity, real_validity)
    g_loss.backward()
    optimizer_G.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{epochs} - D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}")

# =======================
# 4. Generate Synthetic Data
# =======================
def generate_samples(n, label):
    z = torch.randn(n, latent_dim)
    labels = torch.full((n,), label, dtype=torch.long)
    samples = generator(z, labels).detach().numpy()
    return samples

minority_class = y_train.value_counts().idxmin()
n_to_generate = y_train.value_counts().max() - y_train.value_counts().min()
synthetic_data = generate_samples(n_to_generate, minority_class)
synthetic_labels = np.full(n_to_generate, minority_class)

X_aug = np.vstack([X_train_scaled, synthetic_data])
y_aug = np.hstack([y_train, synthetic_labels])

pd.DataFrame(synthetic_data).to_csv("outputs/sample_synthetic_rows.csv", index=False)

# =======================
# 5. Train Classifier
# =======================
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_aug, y_aug)

y_pred = clf.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.savefig("outputs/confusion_matrix.png")
plt.close()

# =======================
# 6. Save Models
# =======================
os.makedirs("models", exist_ok=True)
torch.save(generator.state_dict(), "models/generator_cgan.pt")
joblib.dump(clf, "models/classifier_rf.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Training complete. Models and outputs saved.")
