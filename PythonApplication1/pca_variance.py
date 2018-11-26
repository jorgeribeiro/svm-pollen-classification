import numpy as np
import matplotlib.pyplot as plt
import os

from time import time
from PIL import Image, ImageOps
from sklearn import decomposition, linear_model
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

# allocate dataset
X = np.zeros(4225)
h, w = 65, 65

# allocate target
y = np.zeros((5760,), dtype=np.int32)
for i in range(5760):
	if i < 2880:
		y[i] = 0
	else:
		y[i] = 1

# load data set
function_list = [lambda image:ImageOps.grayscale(image),
				 lambda image:ImageOps.flip(image), lambda image:ImageOps.mirror(image)]

def load_data_set(data):
	for name in os.listdir(os.getcwd()):
		img1 = Image.open(name).crop((0, 0, h, w))
		for	f in function_list:
			img = f(img1)
			for x in range(0, 360, 90):
				img = img.rotate(x)
				data = np.vstack((data, np.array(img).reshape(-1)))

		img2 = Image.open(name).crop((0, 70, 65, 135))
		for	f in function_list:
			img = f(img2)
			for x in range(0, 360, 90):
				img = img.rotate(x)
				data = np.vstack((data, np.array(img).reshape(-1)))
	return data

print("Loading the dataset")
t0 = time()
os.chdir("image_samples/A/")
X = load_data_set(X)
os.chdir("../Ch/")
X = load_data_set(X)
print("done in %0.3fm" % ((time() - t0) / 60))

# remove row with zeros
X = np.delete(X, 0, 0)

# applying PCA
pca = decomposition.RandomizedPCA(whiten=True)
pca.fit(X)
logistic = linear_model.LogisticRegression()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')

n_components = [180, 185, 190, 195, 200, 205, 210, 215, 220, 225]
Cs = np.logspace(-4, 4, 3)

estimator = GridSearchCV(pipe, dict(pca__n_components=n_components, logistic__C=Cs))
estimator.fit(X, y)

plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
plt.show()
