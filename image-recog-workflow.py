import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import pickle

fs = "C:/Users/jackx/Desktop/image-recog/images/cifar-100-python/train"

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

train=unpickle(fs)
print(train)
"""
(train_images, train_labels), (test_images, test_labels) = 
train_images, test_images = train_images / 255.0, test_images/255.0

class_names = ['bird', 'cat', 'deer', 'dog', 'horse']

plt.figure(figsize=(5,5))
for i in range(5):
    plt.subplot(2,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
plt.show()
"""