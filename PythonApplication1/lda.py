"""
==========
SVM
==========

Testing SVM in pollen grains.

Current step: setting up the LDA
"""
print(__doc__)

import numpy as np
import os
from PIL import Image
from sklearn import decomposition
from sklearn import discriminant_analysis

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# change current working directory
os.chdir("image_samples/classifier_A/")

# allocate dataset
data = np.zeros(18225)

# allocate target
y = np.zeros((56,), dtype=np.int32)
for i in range(56):
	if i < 28:
		y[i] = 0
	else:
		y[i] = 1

# load images and put into a matrix [28 x 18225]
for name in os.listdir(os.getcwd()):
	img = Image.open(name).convert("L").crop((0, 0, 135, 135))
	data = np.vstack((data, np.array(img).reshape(-1)))

os.chdir('..')
os.chdir("classifier_Ch/")

for name in os.listdir(os.getcwd()):
	img = Image.open(name).convert("L").crop((0, 0, 135, 135))
	data = np.vstack((data, np.array(img).reshape(-1)))

# remove row with zeros
data = np.delete(data, 0, 0)
print(data)

# setup plot
fig = plt.figure(1, figsize=(6, 5))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.title("LDA of dataset")

# applying LDA
lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=3)
lda.fit(data, y)
data_r = lda.transform(data)
print(data_r) # matriz com 1 componente????

# plotting
# ax.scatter(data_r[:, 0], data_r[:, 1], data_r[:, 2], c='red', cmap=plt.cm.spectral)

# plt.show()