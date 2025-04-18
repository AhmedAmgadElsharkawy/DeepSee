import cv2
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# Open a file dialog to choose the image
root = tk.Tk()
root.withdraw()  # Hide the Tkinter root window
file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg;*.bmp")])

if file_path:
    # Load the chosen image
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)

    # Draw keypoints on the image
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

    # Convert image from BGR to RGB (for displaying with matplotlib)
    image_with_keypoints_rgb = cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB)

    # Display the image using matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(image_with_keypoints_rgb)
    plt.axis('on')  # Show axis for reference
    plt.title('Image with SIFT Keypoints')

    # Show the plot, interactive zooming and panning available
    plt.show()

    # Print the descriptors
    print("Descriptors: \n", descriptors)
else:
    print("No file selected.")
