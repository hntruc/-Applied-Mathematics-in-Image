import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

def data(path):
    img = Image.open(path).convert('RGB')
    img_npa = np.asarray(img)# convert to numpy array

    shape = np.array(img_npa).shape
    width = shape[0]
    height = shape[1]
    number_chanel = shape[2] #number of main colors RGB

    img_1d = np.reshape(img,(width*height, number_chanel))
    return width, height, img_1d

def add_vector(v, w):
    return [vi + wi for vi, wi in zip(v, w)]

def sub_vector(v, w):
    v = np.array(v, dtype = 'int')
    w = np.array(w, dtype = 'int')
    return [vi - wi for vi, wi in zip(v, w)]

def inner_product(v, w):
    return sum(vi*wi for vi, wi in zip(v, w))

def norm_square(v):
    return inner_product(v, v)

def norm(v):
    return math.sqrt(norm_square(v))

def kmeans(img_1d, k_clusters, max_iter, init_centroids='random'):
    # Step 1: get centers
    centers = []
    if init_centroids == 'in_pixels':
        centers = img_1d[np.random.choice(img_1d.shape[0], k_clusters, replace=False)]
    elif init_centroids == 'random':
        temp = []
        while len(temp) < k_clusters:
            ele = []
            for _ in range(3):
                num = random.randint(0,255)
                ele.append(num)
            if ele not in temp:
                temp.append(ele) 
        centers = np.array(temp, dtype = 'uint8')
    
    # Step 2: Calculate distance from each pixel to k centers then pick min value (min value = a group that pixel belongs to), assign label for each pixels.
    temp = img_1d.copy()
    labels = [1 for i in range(img_1d.shape[0])]
    while max_iter > 0:
        for index,element in enumerate(img_1d):
            dis = [norm(sub_vector(i, img_1d[index])) for i in centers]
            labels[index] = np.argmin(dis)

        old_center = [c for c in centers]
        new_old_center = np.array(old_center)

    #Step 3: Calculate new centers
        for ki in range(k_clusters): 
            group = []
            for index,element in enumerate(img_1d):
                if labels[index] == ki:
                    group.append(element)
            new_group = np.array(group)
            avg = np.mean(new_group, axis = 0)
            centers[ki] = avg

    # Compare whether if new centers equal to old center or not. If yes, kmean function will stop and vice versa.
        if (set([tuple(a) for a in new_old_center]) == set([tuple(a) for a in centers])): #check if old_center == new_center
            break 
    
        max_iter-=1

    # Assign new value for each pixel based on label value.
    for l in range(len(labels)):
        for k in range(k_clusters):
            if labels[l] == k:
                temp[l] = centers[k]

    return labels, centers, temp

path = input('Input image: ')
w, h, img_1d = data(path)

k_clusters = input('Input k clusters: ')
k_clusters = int(k_clusters)

max_iter = input('Input max iteration: ')
max_iter = int(max_iter)

init_centroids = input('Input init centroids: ')

labels, centers, matrix = kmeans(img_1d, k_clusters, max_iter, init_centroids='random')
matrix = np.reshape(matrix, (w, h, 3))

imgplot = plt.imshow(matrix)
plt.show()