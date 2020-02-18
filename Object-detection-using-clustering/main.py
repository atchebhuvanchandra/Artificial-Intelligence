import numpy as np
import cv2
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from copy import deepcopy
im = mpimg.imread('im-1.jpg') 

# path to input image specified and  
# image is loaded with imread command 
image = cv2.imread('im-1.jpg') 
# convert the input image into 
# grayscale color space 
operatedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
# modify the data type 
# setting to 32-bit floating point 
operatedImage = np.float32(operatedImage) 
  
# apply the cv2.cornerHarris method 
# to detect the corners with appropriate 
# values as input parameters 
dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07)


# Results are marked through the dilated corners 
dest = cv2.dilate(dest, None) 
  
# Reverting back to the original image, 
# with optimal threshold value 
image[dest > 0.01 * dest.max()]=[0, 0, 255]

# the window showing output image with corners 
cv2.imshow('Image with Borders', image)

#finding top 300 keypoint among the borders selected
corners = cv2.goodFeaturesToTrack(operatedImage,300,0.01,10)
corners = np.int0(corners)
x=list()
y=list()

for i in corners:
    a,b = i.ravel()
    x.append(a)
    y.append(b)
    
fig, ax = plt.subplots()
ax.imshow(im)
ax.plot(x,y, '.r', markersize =3)
ax.axis((0,1280,660,0))
plt.show()

# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

# Number of clusters
k = 4
# X coordinates of random centroids
C_x = np.random.randint(100, 1980, size=k)
# Y coordinates of random centroids
C_y = np.random.randint(300, 660, size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
F = np.array(list(zip(x,y)))
    
# Plotting along with the Centroids
fig , ax = plt.subplots()
ax.imshow(im)
ax.axis((0,1280,660,0))
ax.imshow(im)
fig.suptitle('initialization of centroids',fontsize = 20)
ax.plot(x,y, '.r', markersize =3)
plt.scatter(C_x, C_y, marker='*', s=200, c='white')
plt.show()

# To store the value of centroids when it updates
C_old = np.zeros(C.shape)
# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(F))
# Error func. - Distance between new centroids and old centroids
error = dist(C, C_old, None)
# Loop will run till the error becomes zero
while error != 0:
    # Assigning each value to its closest cluster
    for i in range(len(F)):
             distances = dist(F[i], C)
             cluster = np.argmin(distances)
             clusters[i] = cluster
    # Storing the old centroid values
    C_old = deepcopy(C)
    # Finding the new centroids by taking the average value
    for i in range(k):
      points = [F[j] for j in range(len(F)) if clusters[j] == i]
      C[i] = np.mean(points, axis=0)
    error = dist(C, C_old, None)
colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig , ax = plt.subplots()
ax.axis((0,1280,660,0))
ax.imshow(im)
for i in range(k):
          points = np.array([F[j] for j in range(len(F)) if clusters[j] == i])
          ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='white')
fig.suptitle(' final cluster formations',fontsize = 20)
plt.show()


def bounding_box(points):
    """returns a list containing the bottom left and the top right 
    points in the sequence
    Here, we use min and max four times over the collection of points
    """
    bot_left_x = min(point[0] for point in points)
    bot_left_y = min(point[1] for point in points)
    top_right_x = max(point[0] for point in points)
    top_right_y = max(point[1] for point in points)

    return [(bot_left_x, bot_left_y), (top_right_x, top_right_y)]


fig , ax = plt.subplots()
ax.axis((0,1280,660,0))
colors = [(240,248,255), (118,238,198), (0,255,255),(0,255,0),(255,0,0)]
p = list()
for i in range(k):
          points = np.array([F[j] for j in range(len(F)) if clusters[j] == i])
          p = bounding_box(points)
          c=colors[i]
          im = cv2.rectangle(im,p[0],p[1],c,2)
ax.imshow(im)
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='white')
fig.suptitle('Bounding box',fontsize = 20)
plt.show()
