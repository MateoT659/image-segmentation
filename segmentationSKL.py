# Kmeans clustering for image segmentation using sklearn

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def show_image(img, title):
    # Helper function to display an image
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()

# Getting image path and processing image

path = input("Enter the path to the image: ")

while not os.path.exists(path):
    print("Image not found. Please enter a valid path.")
    path = input("Enter the path to the image: ")

file = Image.open(path)
img = np.array(file)
show_image(img, "Original Image")

# Getting number of clusters

n_clusters = input("Enter the number of clusters: ")
n_clusters = int(n_clusters) if n_clusters.isdigit() else 3

# Clustering image

X = img.reshape(-1, 3)
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

Xnew = [[int(k) for k in kmeans.cluster_centers_[i]] for i in kmeans.labels_]
Xnew = np.array(Xnew).reshape(-1, 3)
imgseg = np.array(Xnew).reshape(img.shape)

# Display and save segmented image

show_image(imgseg, "Segmented Image with " + str(n_clusters) + " Clusters")

filename = path.split(".")[0].split("/")[-1]
plt.imsave(f"output/{filename}_{str(n_clusters)}_cluster.png", imgseg.astype(np.uint8))
print("Segmented image saved")