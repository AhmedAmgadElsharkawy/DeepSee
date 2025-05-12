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
            img = cv2.resize(img, image_size)
            img = img.astype(np.float32) / 255.0
            faces.append(img.flatten().reshape(-1, 1))
    return np.array(faces)

# PCA setup and face recognition
def face_recognition_process(test_img, lowe_ratio, pca_confidence_level, training_faces):
    X = training_faces.reshape((training_faces.shape[0], -1)).T
    mean_face = np.mean(X, axis=1, keepdims=True)
    X_centered = X - mean_face

    L = X_centered.T @ X_centered
    eigvals, eigvecs = np.linalg.eigh(L)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    total_variance = np.sum(eigvals)
    cumulative_variance = np.cumsum(eigvals / total_variance)
    k = np.searchsorted(cumulative_variance, pca_confidence_level) + 1

    eigenfaces = X_centered @ eigvecs
    eigenfaces = eigenfaces / np.linalg.norm(eigenfaces, axis=0)
    eigenfaces = eigenfaces[:, :k]

    projections = eigenfaces.T @ X_centered

    test_centered = handle_test_image(test_img, mean_face)
    test_proj = eigenfaces.T @ test_centered
    distances = np.linalg.norm(projections - test_proj, axis=0)
    distance_with_indices = np.column_stack((distances, np.arange(len(distances))))
    distance_with_indices = distance_with_indices[distance_with_indices[:, 0].argsort()]

    best_distance, best_match = distance_with_indices[0]
    best_image_class = int(best_match) // 10

    matched = False
    for i in range(1, 11):  # check first 10 nearest neighbors
        other_class = int(distance_with_indices[i][1]) // 10
        if other_class != best_image_class:
            if best_distance < lowe_ratio * distance_with_indices[i][0]:
                matched = True
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
            print(subject_id)
            images.append(img)
            labels.append(subject_id)
    return images, labels

# Main script to evaluate and plot ROC curves
if __name__ == "__main__":
    training_faces = load_training_data(dataset_path)
    test_images, test_labels = load_test_images(test_path)

    all_lowes = [0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 1.0]
    plt.figure(figsize=(10, 7))

    for lowe_ratio in all_lowes:
        y_scores = []
        y_true = []

        for img, true_label in zip(test_images, test_labels):
            matched, predicted_class, best_distance = face_recognition_process(img, lowe_ratio, pca_confidence_level, training_faces)
            print(f"matched: {matched}, predicted_class: {predicted_class}, true_label: {true_label}, best_distance: {best_distance}")
            if matched:
                y_scores.append(1 / (1 + best_distance))
                # Store true label (1 if correct match)
                y_true.append(1 if predicted_class == true_label // 10 else 0)

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
