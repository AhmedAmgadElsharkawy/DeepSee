import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

dataset_path = "data/train_faces"
test_path = "data/test_faces"
image_size = (64, 64)
pca_confidence_level = 0.95

# Load training images
def load_training_data(path):
    faces = []
    for filename in os.listdir(path):
        if filename.lower().endswith('.png'):
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # img = cv2.resize(img, image_size)
            img = img.astype(np.float32) / 255.0
            faces.append(img.flatten().reshape(-1, 1))
    return np.array(faces)

# PCA setup and face recognition
def face_recognition_process(test_img, lowe_ratio, pca_confidence_level):
    faces = []
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith('.png'):
            img_path = os.path.join(dataset_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32) / 255.0
            faces.append(img.flatten().reshape(-1, 1))
    
    faces = np.array(faces)
    X = faces.reshape((faces.shape[0], -1)).T
    mean_face = np.mean(X, axis=1, keepdims=True)
    X_centered = X - mean_face

    L = X_centered.T @ X_centered
    eigvals, eigvecs = np.linalg.eigh(L)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    total_variance = np.sum(eigvals)
    variance_ratio = eigvals / total_variance
    cumulative_variance = np.cumsum(variance_ratio)
    k = np.searchsorted(cumulative_variance, pca_confidence_level) + 1

    eigenfaces = X_centered @ eigvecs
    eigenfaces = eigenfaces / np.linalg.norm(eigenfaces, axis=0)
    eigenfaces = eigenfaces[:, :k]

    projections = eigenfaces.T @ X_centered

    test_centered = handle_test_image(test_img, mean_face)
    test_proj = eigenfaces.T @ test_centered
    distances = np.linalg.norm(projections - test_proj, axis=0)
    distance_with_indices = []

    label = 0

    for idx, dist in enumerate(distances):
        while str(label).endswith('9'):
            label += 1

        distance_with_indices.append([dist, label])
        label += 1

    distance_with_indices = np.array(distance_with_indices)        
    distance_with_indices = distance_with_indices[distance_with_indices[:, 0].argsort()]

    best_distance, best_match = distance_with_indices[0]

    best_image_class = best_match // 10
    matched = None
    
    for i in range(1, 11):  # check first 10 nearest neighbors
        other_class = distance_with_indices[i][1] // 10
        if other_class != best_image_class:
            if best_distance < lowe_ratio * distance_with_indices[i][0]:
                matched = True
            else: 
                best_image_class = other_class
            break
    return matched, best_image_class, best_distance

def handle_test_image(test_img, mean_face):
    if test_img.dtype != np.float32 and test_img.max() > 1:
        test_img = test_img.astype(np.float32) / 255.0
    if test_img.shape != image_size:
        test_img = cv2.resize(test_img, image_size)
    return test_img.flatten().reshape(-1, 1) - mean_face

# Load test images and ground truth labels
def load_test_images(test_path):
    images = []
    labels = []
    for filename in os.listdir(test_path):
        if filename.lower().endswith('.png'):
            img_path = os.path.join(test_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, image_size)
            subject_id = int(filename.split('_')[1].split('.')[0])
            images.append(img)
            labels.append(subject_id)
    return images, labels

# Main script to evaluate and plot ROC curves
if __name__ == "__main__":
    training_faces = load_training_data(dataset_path)
    test_images, test_labels = load_test_images(test_path)

    all_lowes = [0.4, 0.5, 0.6, 0.7]
    plt.figure(figsize=(10, 7))

    for lowe_ratio in all_lowes:
        y_scores = []
        y_true = []

        for img, true_label in zip(test_images, test_labels):
            matched, predicted_class, best_distance = face_recognition_process(img, lowe_ratio, pca_confidence_level)
            

            y_scores.append(1 / (1 + best_distance))
            y_true.append(1 if predicted_class == (true_label // 10) else 0)

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Lowe Ratio={lowe_ratio:.2f}, AUC={auc_score:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Face Recognition at Different Lowe Ratios")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
