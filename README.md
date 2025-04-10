# Image Segmentation Study

## Sklearn Implementation
```segmentationSKL.py``` contains code that uses sklearn to perform kmeans clustering on image data.

To run this model, please use ```python segmentationSKL.py``` in the command line, then provide a filename and the number of clusters desired.

## Custom Implementation

```segmentation.py``` contains a custom implementation of kmeans clustering and mean shift clustering on image data.

To run this model, please use ```python segmentation.py``` in the command line.

Then, provide a filename, which algorithm to use (KMeans, Mean Shift), how many clusters to use (if KMeans is picked), and what color space to use (RGB, YUV, LAB).

Note that this implementation uses max pooling to reduce the size of the image before segmenting for efficiency. This may erase some colors that do not occur frequently enough in the image.

### Color Spaces:

- RGB (default): Basic red-green-blue color scheme. (Red, Green, Blue)

- YUV: Separates luminance from color information. Reduces color resolution, but is good for classifying images with subtle brightnesses. (Luma, U, V)

- LAB: Uniformly distributes colors based on the 2-norm. Provides strong contrast detection. (Light, Green-Red, Blue-Yellow)

## DEPENDENCIES:

numpy:
```bash
pip install numpy
```

matplotlib:
```bash
pip install matplotlib
```

Pillow:
```bash
pip install Pillow
```

scikit-image:
```bash
pip install scikit-image
```

scikit-learn:
```bash
pip install scikit-learn
```
