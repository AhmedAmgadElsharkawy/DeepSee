import cv2
import numpy as np
from sklearn.cluster import KMeans
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
            
            pass
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
            
            center = new_centers
        
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

        pixels = image.reshape((-1, 3)).astype(int)

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
            i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)

            merged = np.vstack((clusters[i], clusters[j]))
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
            center = np.mean(cluster, axis=0)
            centers.append(center)
            for point in cluster:
                lookup[tuple(np.round(point).astype(int))] = idx
        return lookup, centers