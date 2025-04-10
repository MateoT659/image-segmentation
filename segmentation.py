# Kmeans and mean shift clustering for image segmentation using custom implementation

# IMPORTS ------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import time
from skimage.color import rgb2lab, lab2rgb, rgb2yuv, yuv2rgb
from sklearn.cluster import estimate_bandwidth

def show_image(img, title):
    # Helper function to display an image using matplotlib
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()

# GETTING PARMETERS ---------------------------------------------------------------

# Image path
path = input("Enter the path to the image: ")

while not os.path.exists(path):
    print("Image not found. Please enter a valid path.")
    path = input("Enter the path to the image: ")

file = Image.open(path)
file.convert("RGB")
img = np.array(file)


# Algorithm select
algorithm = input("Enter the algorithm to be used (KMeans, Mean Shift): ").strip().upper()

if algorithm not in ["KMEANS", "MEAN SHIFT"]:
    print("Invalid algorithm. Using KMeans as default.")
    algorithm = "KMEANS"

if algorithm == "KMEANS":
    # Number of clusters for KMeans
    n_clusters = input("Enter the number of clusters: ").strip()
    if not n_clusters.isdigit():
        print("Invalid input. Using 3 clusters as default.")
        n_clusters = 3
    else:
        n_clusters = int(n_clusters)

# Color space select
colorspace = input("Enter the color space (RGB, YUV, LAB): ").strip().upper()

if colorspace not in ["RGB", "YUV", "LAB"]:
    print("Invalid color space. Using RGB as default.")
    colorspace = "RGB"

# IMAGE PROCESSING ---------------------------------------------------------------

# Compressing image using max pooling
print("\nPerforming max pooling on the image...")

def pool(img, pool_size=2):
    # Performs max pooling on the image
    pooled_img = np.zeros((img.shape[0] // pool_size, img.shape[1] // pool_size, img.shape[2]), dtype=img.dtype)
    for i in range(0, img.shape[0]-pool_size+1, pool_size):
        for j in range(0, img.shape[1]-pool_size+1, pool_size):
            pooled_img[i // pool_size, j // pool_size] = np.max(img[i:i + pool_size, j:j + pool_size], axis=(0, 1))

    return pooled_img

pooledimg = img.copy()
poolcount = 0

if algorithm == "MEAN SHIFT":
    poolthresh = 10000  
else:
    poolthresh = 200000  
    
while pooledimg.shape[0]*pooledimg.shape[1] > poolthresh:
    pooledimg = pool(pooledimg, 2)
    poolcount += 1

print(f"Image pooled {poolcount} times\n")
show_image(img, "Original Image")
show_image(pooledimg, "Pooled Image")

# Reshaping and converting color space

X = pooledimg.reshape(-1, 3)
Xtrain = X.copy()

if colorspace == "RGB":
    Xtrain = Xtrain
elif colorspace == "YUV":
    Xtrain = rgb2yuv(Xtrain)
elif colorspace == "LAB":
    Xtrain = rgb2lab(Xtrain)  
else:
    # default to RGB if invalid color space
    Xtrain = Xtrain  

centers = []

# CLUSETERING ---------------------------------------------------------------

print("Performing clustering...")

if algorithm == "MEAN SHIFT":
    start_time = time.time()
    # MEAN SHIFT CLUSTERING -------------------------------------------------

    bandwidth = estimate_bandwidth(Xtrain, quantile=0.1)  # estimate bandwidth using sklearn

    def within_bandwidth(x, center, bandwidth):
        # check if a point x is within a distance of the center
        return np.linalg.norm(x - center) < bandwidth

    Xtrain = np.unique(Xtrain, axis=0)


    centermap = {}
    for i, test in enumerate(Xtrain):
        # for each point, find mean shift center
        prevCenter = None
        currentCenter = test

        while prevCenter is None or np.linalg.norm(currentCenter - prevCenter) > 0.01:
            prevCenter = currentCenter
            within_band = Xtrain[np.linalg.norm(Xtrain - currentCenter, axis=1) < bandwidth]
            currentCenter = np.mean(within_band, axis=0)
        if not any(np.all(currentCenter == center) for center in centers):
            centermap[tuple(currentCenter)] = centermap[tuple(currentCenter)]+1 if tuple(currentCenter) in centermap else 1
        
        if i%120 == 0:
            print(f"Mean shift iteration {i/len(Xtrain) *100:.2f}% completed")

    if len(centermap) > 60:
        centermap = dict(sorted(centermap.items(), key=lambda item: item[1], reverse=True)[:60])
    centers = np.array(list(centermap.keys()))
    print(f"Mean shift clustering completed in {time.time() - start_time:.2f} seconds")
    print(f"Number of clusters found: {len(centers)}")
else:

    # KMEANS CLUSTERING -----------------------------------------------------

    centers = [np.random.randint(0, 256, size=(3)) for _ in range(n_clusters)]
    centersPrev = np.zeros((n_clusters, 3))
    tol = 1e-4
    error = sum(np.linalg.norm(centers[i] - centersPrev[i]) for i in range(n_clusters))

    max_iter = 100
    i = 0

    max_time = 120
    start_time = time.time()

    while error > tol and i < max_iter and (time.time() - start_time) < max_time:
        distances = np.zeros((Xtrain.shape[0], n_clusters))
        for j in range(n_clusters):
            distances[:, j] = np.linalg.norm(Xtrain - centers[j], axis=1)

        labels = np.argmin(distances, axis=1)  # find closest center for each pixel

        centersPrev = centers.copy()  # store previous centers
        counts = [0] * n_clusters
        centers = np.zeros((n_clusters, 3))  # reset centers

        for j in range(len(Xtrain)):
            counts[labels[j]] += 1
            centers[labels[j]] += Xtrain[j]
        
        centers = np.array([centers[j] / counts[j] if counts[j] > 0 else centers[j] for j in range(n_clusters)])  # update centers
        error = sum(np.linalg.norm(centers[i] - centersPrev[i]) for i in range(n_clusters))  # calculate error
        i += 1
        if i % 10 == 0:
            print(f"Iteration {i}, Time: {time.time()-start_time:.2f}s")
    
    print(f"Image segmented in {i} iterations (Average time per iteration: {(time.time()-start_time)/i:.2f}s)")

# POST-PROCESSING ---------------------------------------------------------------

# Convert back to RGB

print("\nProcessing segmented image...")

if colorspace == "YUV":
    centers = yuv2rgb(centers)  
    centers = np.clip(centers, 0, 1) * 255 
elif colorspace == "LAB":
    centers = lab2rgb(centers) 
    centers = np.clip(centers, 0, 1) * 255 

centers = [[int(k) for k in c] for c in centers] 
centers = np.array(centers)

# Assign each pixel to the closest center

X = img.reshape(-1, 3)
labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1) 

Xnew = [centers[i] for i in labels]
Xnew = np.array(Xnew).reshape(-1, 3)
imgseg = np.array(Xnew).reshape(img.shape)

# Display and save segmented image

filename = path.split(".")[0].split("/")[-1]

if algorithm == "MEAN SHIFT":
    show_image(imgseg, "Segmented Image using Mean Shift using " + colorspace)
    filename = f"{filename}_meanshift_{colorspace}"
else:
    show_image(imgseg, "Segmented Image with " + str(n_clusters) + " Clusters using " + colorspace)
    filename = f"{filename}_kmeans_{str(n_clusters)}_{colorspace}"

plt.imsave(f"output/{filename}.png", imgseg.astype(np.uint8))
print("Segmented image saved at output/" + filename + ".png")