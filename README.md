# DeepSee
DeepSee is an advanced image processing and analysis project that explores various spatial and frequency domain techniques to enhance, manipulate, and interpret images. The project is structured into multiple phases, covering essential aspects of image processing, from basic filtering to sophisticated facial detection and recognition. With cutting-edge algorithms, DeepSee efficiently processes diverse image types, including biometric images and facial profiles, to extract meaningful insights and achieve enhanced visual representations.

## Table of Contents
- [Demo](#demo)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Features](#features)
- [Contributors](#contributors)

## Demo
https://github.com/user-attachments/assets/0b4b47f7-5114-4c16-b28f-24f219135f5d

## Prerequisites

- Python 3.6 or higher

## Installation

1. **Clone the repository:**

   ``````
   git clone https://github.com/AhmedAmgadElsharkawy/DeepSee.git
   ``````

2. **Install The Dependincies:**
    ``````
    pip install -r requirements.txt
    ``````

3. **Run The App:**

    ``````
    python main.py
    ``````

## Features  
- **Spatial and Frequency Domain Processing**:  
  Apply filters for noise reduction, edge enhancement, and feature extraction. Perform histogram equalization, image normalization, thresholding, and contrast enhancement on various image types.  

- **Hybrid Images**:  
  Construct hybrid images using Laplacian pyramids, blending low-frequency contents of one image with the high-frequency contents of another for seamless visual effects.  

- **Boundary Detection and Representation**:  
  Detect boundaries using active contours and represent them through chain codes, meshes, polygons, and object skeletons. Includes 2D, 3D, and 4D measurements of length, area, surface, and volume.  

- **Edge Detection and Hough Transform**:  
  Identify edges and boundaries efficiently using multiple edge-detection techniques, and detect shapes like lines, circles, and ellipses using the Hough Transform.  

- **Feature Detection and Image Matching**:  
  Extract feature points and generate robust descriptors using SIFT. Implement image matching using Sum of Squared Differences (SSD) and normalized cross-correlation for precise alignment.  

- **Image Segmentation**:  
  Perform both global and local thresholding (Otsu's method, spectral thresholding) and unsupervised segmentation techniques, including k-means, region growing, agglomerative clustering, and mean shift.  

- **Face Detection and Recognition**:  
  Detect and recognize faces using Eigen analysis, extending functionality to facial expression analysis with high accuracy.  

- **Optimized GUI with Multiprocessing and Multithreading**:  
  Seamlessly handle multiple operations without freezing or lagging the interface. Image processing tasks are executed in parallel using multiprocessing, while GUI responsiveness is maintained through multithreading.  

- **Dark and Light Modes**:  
  Choose between a sleek dark mode and a clean light mode for comfortable viewing in any environment.  

- **Responsive Design**:  
  The application layout adapts gracefully to different screen sizes, ensuring a smooth user experience on various devices.  


## Contributors
- **AhmedAmgadElsharkawy**: [GitHub Profile](https://github.com/AhmedAmgadElsharkawy)
- **AbdullahMahmoudHanafy**: [GitHub Profile](https://github.com/AbdullahMahmoudHanafy)
- **somaiaahmed**: [GitHub Profile](https://github.com/somaiaahmed)
- **PavlyAwad**: [GitHub Profile](https://github.com/PavlyAwad)

