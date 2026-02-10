## Methodology

### Dataset Collection
The dataset consists of flower images captured around a college campus under varying lighting conditions and backgrounds. The dataset includes both close-up images of single flowers and images containing clusters of flowers and surrounding foliage.

### Image Standardization
All raw images were converted to a uniform format (JPEG) and renamed using a consistent naming scheme to ensure reproducibility and ease of processing.

### Image Preprocessing
Images were resized to a fixed resolution and subjected to Gaussian smoothing to reduce noise. Color space conversion was performed to represent images in HSV format for improved color analysis.

### Segmentation
Flower regions were segmented from the background using the GrabCut algorithm. Morphological operations were applied to refine the segmentation masks and reduce background artifacts.

### Feature Extraction
For each segmented image, the following features were extracted:
- Mean Hue, Saturation, and Value (color features)
- Area ratio, perimeter, and circularity (shape features)
- GLCM-based texture contrast (texture feature)

### Feature Preprocessing
Invalid samples resulting from segmentation failures were removed. All numerical features were standardized to zero mean and unit variance to prepare the data for distance-based clustering.

### Clustering and Visualization
K-Means clustering was applied to the preprocessed feature vectors. The elbow method was used to select an appropriate number of clusters. Principal Component Analysis (PCA) was used to visualize the clustering results in two dimensions.
