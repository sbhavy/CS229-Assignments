import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def init_centroids(num_clusters, image):

    H, W, _ = image.shape
    rand_num = np.random.randint((H*W), size = num_clusters)
    return image[rand_num // W, rand_num % W].astype(float)

def update_centroid(centroids, image):

    H, W, _ = image.shape
    temp = np.zeros((H,W))
    for i in range(30):
        for j in range(H):
            for k in range(W):
                temp[j][k] = np.argmin(np.linalg.norm(centroids-image[j][k]), axis = 0)

        for j in range(centroids.shape[0]):
            pixels = image[temp==j]
            if pixels.shape[0] > 0:
                centroids[j] = pixels.mean(axis=0)
    
    return centroids.astype(int)

def update_image(centroids, image):

    H, W, _ = image.shape
    for i in range(H):
        for j in range(W):
            image[i][j] = centroids[np.argmin(np.linalg.norm(centroids - image[i][j], axis = 1))]

    return image

image = mpimg.imread('../data/peppers-small.tiff')
centroids_init = init_centroids(16, image)
centroids = update_centroid(centroids_init, image)

image = np.copy(mpimg.imread('../data/peppers-large.tiff'))
modified_image = update_image(centroids, image)
plt.figure()
plt.imshow(modified_image)
plt.savefig('../data/peppers-modified.tiff')