import cv2
import numpy as np
from sklearn.cluster import KMeans
import time
from collections import defaultdict
from scipy.spatial.distance import cdist

class SegmentationController():
    def __init__(self,segmentation_window = None):
        self.segmentation_window = segmentation_window
        if self.segmentation_window:
            self.segmentation_window.apply_button.clicked.connect(self.apply_segmentation)


    def apply_segmentation(self):
        image = self.segmentation_window.input_image_viewer.image_model.get_image_matrix()
        k_value = self.segmentation_window.k_means_k_value_spin_box.value()
        max_iterations = self.segmentation_window.k_means_max_iterations_spin_box.value()
        
        
        region_threshold = self.segmentation_window.region_growing_threshold_spin_box.value()
        markers = [(m['x'], m['y']) for m in self.segmentation_window.input_image_viewer.markers_positions]
        
        if self.segmentation_window.segmentation_algorithm_custom_combo_box.current_text() == "k-means":
            output_image = self.k_means_segmentation(image, k_value, max_iterations)
        elif self.segmentation_window.segmentation_algorithm_custom_combo_box.current_text() == "Mean Shift":
            spatial_radius=self.segmentation_window.mean_shift_spatial_radius_spin_box.value()
            color_radius=self.segmentation_window.mean_shift_color_radius_spin_box.value()
            
            output_image=self.mean_shift_segmentation(image,spatial_radius=spatial_radius,color_radius=color_radius)
        elif self.segmentation_window.segmentation_algorithm_custom_combo_box.current_text() == "Agglomerative Segmentation":
            cluster_numbers=self.segmentation_window.agglomerative_segmentation_clusters_number_spin_box.value()
            initial_ccluster_numbers= self.segmentation_window.agglomerative_segmentation_initial_clusters_number_spin_box.value()

            output_image=self.agglomerative_segmention(image,cluster_numbers,initial_ccluster_numbers)

           

        else:
           
            output_image = self.region_growing_segmentation(image, markers, region_threshold)   
                      
        self.segmentation_window.output_image_viewer.display_and_set_image_matrix(output_image)


    def k_means_segmentation(self, image, k_value, max_iterations):
        # Reshape the image to a 2D array of pixels
        pixels = image.reshape((-1, 3)).astype(np.float32)
        num_pixels = pixels.shape[0]
        
        indices = np.random.choice(num_pixels, k_value, replace=False)
        centers = pixels[indices]
        
        for _ in range(max_iterations):
            # Calculate distances from each pixel to each center
            distances = np.linalg.norm(pixels[:, np.newaxis] - centers, axis=2)
            
            # Assign each pixel to the nearest center
            labels = np.argmin(distances, axis=1)
            
            # Update centers
            new_centers = np.array([pixels[labels == i].mean(axis=0) if np.any(labels == i) else centers[i]
                                    for i in range(k_value)])            
            # Check for convergence (if centers do not change)
            if np.all(centers == new_centers):
                break
            
            centers = new_centers
        
        # map each pixel to its corresponding color
        segmented_image = centers[labels].reshape(image.shape).astype(np.uint8)
       
        return segmented_image


    def region_growing_segmentation(self, image, markers, threshold):
     
        # Make a copy of the original image
        result = image.copy()
        
        # Convert to grayscale for intensity comparison
        if len(image.shape) == 3 and image.shape[2] >= 3:
            working_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            working_image = image.copy()
        
        h, w = working_image.shape[:2]
        
        # 4-connectivity neighbors (up, down, left, right)
        neighbors = [(0,1), (1,0), (0,-1), (-1,0)]
        
        # Process each seed point
        for seed in markers:
            x, y = int(seed[0]), int(seed[1])
            
            # Validate seed point
            if x < 0 or y < 0 or x >= w or y >= h:
                continue
                
            # Initialize queue and visited matrix
            queue = [(x, y)]
            visited = np.zeros((h, w), dtype=bool)
            seed_value = working_image[y, x]
            
            while queue:
                cx, cy = queue.pop(0)
                
                # Skip if already visited or out of bounds
                if (cx < 0 or cy < 0 or cx >= w or cy >= h or 
                    visited[cy, cx]):
                    continue
                    
                # Check intensity difference
                current_value = working_image[cy, cx]
                if abs(int(current_value) - int(seed_value)) <= threshold:
                    # Mark as visited
                    visited[cy, cx] = True
                    
                    # Handle both 3-channel and 4-channel images
                    if result.shape[2] == 4:  # RGBA
                        result[cy, cx] = [0, 0, 255, 255]  # Red with full alpha
                    else:  # RGB
                        result[cy, cx] = [0, 0, 255]  # Red in BGR
                    
                    # Add neighbors to queue
                    for dx, dy in neighbors:
                        queue.append((cx + dx, cy + dy))
        
        return result

    def agglomerative_segmention(self,image, final_k=10, initial_k=25):

        pixels = image.reshape((-1, 3)).astype(int) #5d array

        # Step 1: Initial Clustering with KMeans
        initial_clusters = self.initial_kmeans_quantization(pixels, initial_k)

        # Step 2: Agglomerative Merging
        final_clusters = self.agglomerative_merge(initial_clusters, final_k)

        # Step 3: Recoloring based on final clusters
        cluster_map, centers = self.build_cluster_lookup(final_clusters)
        recolored = np.array([
            centers[cluster_map.get(tuple(np.round(p).astype(int)), 0)]
            for p in pixels
        ], dtype=np.uint8)

        # Reshape to original image dimensions
        h, w = image.shape[:2]
        segmented = recolored.reshape((h, w, 3))

        return segmented


    def initial_kmeans_quantization(self,points, initial_k):
        """
        Partitions points into `initial_k` groups using KMeans quantization.
        """
        print("[INFO] Performing KMeans-based initial clustering...")
        kmeans = KMeans(n_clusters=initial_k, n_init=5, random_state=42)
        labels = kmeans.fit_predict(points)

        groups = [[] for _ in range(initial_k)]
        for point, label in zip(points, labels):
            groups[label].append(point)

        return [np.array(g) for g in groups if len(g) > 0]


    def agglomerative_merge(self,clusters, target_k):
        """
        Agglomeratively merges clusters until only `target_k` clusters remain.
        """
        print("[INFO] Merging clusters...")
        while len(clusters) > target_k:
            centroids = np.array([np.mean(cluster, axis=0) for cluster in clusters])
            dist_matrix = cdist(centroids, centroids)
            np.fill_diagonal(dist_matrix, np.inf)
            i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)  #Finds the indices (i,j) of the closest pair of clusters in the distance matrix.

            merged = np.vstack((clusters[i], clusters[j])) #Combines the two closest clusters by vertically stacking their pixel arrays.
            clusters[i] = merged
            del clusters[j]

            print(f"[INFO] Clusters remaining: {len(clusters)}")
        return clusters


    def build_cluster_lookup(self,clusters):
        """
        Maps each rounded pixel to its cluster center.
        """
        print("[INFO] Assigning clusters to points...")
        lookup = {}
        centers = []
        for idx, cluster in enumerate(clusters):
            center = np.mean(cluster, axis=0) #averaging all its pixel values along the color channels
            centers.append(center)
            for point in cluster:
                lookup[tuple(np.round(point).astype(int))] = idx
        return lookup, centers

    def mean_shift_segmentation(self,image, spatial_radius=15, color_radius=20, min_shift=0.5, max_iterations=20):
        """
        implementation of Mean Shift segmentation

        """
        # Converts to CIELAB color space which better represents human color perception
        # Convert to CIELAB color space
        scale_percent = 50  # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        image = cv2.resize(image, (width, height))
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Start timing
        start_time = time.time()

        # Reshape the image to a feature vector [x, y, L, a, b]
        height, width, _ = lab_image.shape
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)


        #Combines spatial (x,y) and color (L,a,b) information into a 5D feature vector for each pixel
        pixels = np.column_stack((
            X.flatten(),              # X coordinate
            Y.flatten(),              # Y coordinate
            lab_image[:, :, 0].flatten(),  # L channel
            lab_image[:, :, 1].flatten(),  # a channel
            lab_image[:, :, 2].flatten()   # b channel
        ))

        # Normalize spatial and color feature spaces separately
        spatial_scale = 1.0 / spatial_radius
        color_scale = 1.0 / color_radius

        # Apply scaling to the feature space
        scaled_pixels = np.column_stack((
            pixels[:, 0] * spatial_scale,
            pixels[:, 1] * spatial_scale,
            pixels[:, 2] * color_scale,
            pixels[:, 3] * color_scale,
            pixels[:, 4] * color_scale
        ))

        # Create a working copy
        points = scaled_pixels.copy()

        # Use a grid-based approach to reduce the number of seed points
        grid_size = 20  # Adjust based on your needs
        seed_indices = []

        # Create a grid and select one point from each cell
        for y in range(0, height, grid_size):
            for x in range(0, width, grid_size):
                # Calculate pixel indices in this grid cell
                y_max = min(y + grid_size, height)
                x_max = min(x + grid_size, width)

                # Get a representative point from this cell
                cell_indices = []
                for i in range(y, y_max):
                    for j in range(x, x_max):
                        if i < height and j < width:
                            cell_indices.append(i * width + j)

                if cell_indices:
                    seed_indices.append(cell_indices[len(cell_indices) // 2])  # Middle point

        print(f"Number of seed points: {len(seed_indices)}")

        # Start with seed points
        print("Performing mean shift...")
        seed_points = points[seed_indices].copy()
        shifted_points = seed_points.copy()

        # Mean shift iteration
        for iteration in range(max_iterations):
            max_shift_distance = 0

            for i in range(len(seed_points)):
                # Compute distances from current seed point to all scaled pixels
                distances = np.sqrt(np.sum((points - shifted_points[i])**2, axis=1))

                # Find points within the bandwidth (points where distance <= 1.0)
                in_bandwidth_indices = np.where(distances <= 1.0)[0]

                if len(in_bandwidth_indices) > 0:
                    # Compute mean shift vector
                    new_point = np.mean(points[in_bandwidth_indices], axis=0)

                    # Calculate shift distance
                    shift_distance = np.sqrt(np.sum((new_point - shifted_points[i])**2))
                    max_shift_distance = max(max_shift_distance, shift_distance)

                    # Update position
                    shifted_points[i] = new_point

            print(f"Iteration {iteration+1}, max shift: {max_shift_distance:.4f}")

            # Check convergence
            if max_shift_distance < min_shift:
                print(f"Converged after {iteration+1} iterations")
                break

        # Merge close modes/clusters
        merge_distance = 1.0

        # Dictionary to store clusters
        clusters = {}
        cluster_id = 0

        # First cluster center
        clusters[cluster_id] = shifted_points[0]
        cluster_id += 1

        # Assign shifted points to clusters or create new ones
        for i in range(1, len(shifted_points)):
            # Find closest existing cluster
            min_dist = float('inf')
            closest_cluster = -1

            for cid, center in clusters.items():
                dist = np.sqrt(np.sum((shifted_points[i] - center)**2))
                if dist < min_dist:
                    min_dist = dist
                    closest_cluster = cid

            # If close enough to existing cluster, merge
            if min_dist <= merge_distance:
                # Update cluster center as the weighted average
                clusters[closest_cluster] = (clusters[closest_cluster] + shifted_points[i]) / 2
            else:
                # Create new cluster
                clusters[cluster_id] = shifted_points[i]
                cluster_id += 1

        print(f"Number of clusters after merging: {len(clusters)}")

        # Assign each pixel to the nearest cluster
        cluster_centers = np.array(list(clusters.values()))
        labels = np.zeros(len(points), dtype=int)

        # Use efficient vectorized operations for assignment
        # Calculate distances to all cluster centers
        distances = cdist(points, cluster_centers)

        # Assign each point to the nearest cluster
        labels = np.argmin(distances, axis=1)

        # Create segmented image
        result = np.zeros_like(image)

        # Map each pixel to its cluster center's color
        for i in range(len(clusters)):
            mask = labels == i
            # Get the cluster center
            center = cluster_centers[i]

            # Convert back to original scale
            color_center = np.array([
                center[2] / color_scale,  # L
                center[3] / color_scale,  # a
                center[4] / color_scale   # b
            ])

            # Reshape the mask to the image shape
            mask_2d = mask.reshape(height, width)

            # Set the color of the segment
            # First convert center color from LAB to BGR
            lab_color = np.uint8([[color_center]])
            bgr_color = cv2.cvtColor(lab_color, cv2.COLOR_LAB2BGR)[0][0]

            # Apply the color to the segment
            result[mask_2d] = bgr_color

        # End timing
        end_time = time.time()
        print(f"Custom segmentation completed in {end_time - start_time:.2f} seconds")

        return result