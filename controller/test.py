import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix

# Load dataset
dataset_path = "data/eigen_faces_dataset/olivetti_faces.npy"
faces = np.load(dataset_path)
X = faces.reshape((faces.shape[0], -1)).T  # Shape: (4096, 400) for 64x64 images
y = np.repeat(np.arange(36), 10)  # Labels (40 classes, 10 samples each)
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# Train PCA
pca = PCA(n_components=0.95, whiten=True)  # Keeps 95% variance
X_pca = pca.fit_transform(X.T)  # Shape: (400, n_components)

# Build recognizer
knn = NearestNeighbors(n_neighbors=2, metric='euclidean')
knn.fit(X_pca)

# Generate test scores (example: use 20% holdout set)
X_train, X_test, y_train, y_test = train_test_split(X.T, y, test_size=0.2, stratify=y)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
knn.fit(X_train_pca)

# Get decision scores (1 / distance to closest neighbor)
y_scores = []
for img in X_test:
    img_pca = pca.transform(img.reshape(1, -1))
    distances, _ = knn.kneighbors(img_pca)
    y_scores.append(1 / (1 + distances[0][0]))  # Convert distance to score

# Binarize labels (1 if correct match, 0 otherwise)
y_true = (y_test == y_train[knn.kneighbors(pca.transform(X_test))[1][:, 0]]).astype(int)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Face Recognition')
plt.legend()
plt.show()