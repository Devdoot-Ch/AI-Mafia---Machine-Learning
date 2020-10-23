import cv2
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('elephant.jpg')
cv2.imshow('image', image)
cv2.waitKey(0)

image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
plt.imshow(image_copy)
plt.show()
print(image_copy.shape)
pixels = image_copy.reshape((330*500, 3))
k = 10
km = KMeans(n_clusters=k)
km.fit(pixels)

centers = km.cluster_centers_
centers = np.array(centers, dtype='uint8')
labels = km.labels_

new_image = np.zeros((330*500, 3))
new_image = np.array(new_image, dtype='uint8')
for i in range(new_image.shape[0]):
    new_image[i] = centers[labels[i]]

new_image = new_image.reshape(image_copy.shape)
plt.imshow(new_image)
plt.show()
