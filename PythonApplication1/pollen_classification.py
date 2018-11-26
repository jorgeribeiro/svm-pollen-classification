"""
==========
SVM
==========

Testing SVM in pollen grains.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import os

from time import time
from PIL import Image, ImageOps
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split

# allocate dataset
X = np.zeros(4225)
h, w = 65, 65
target_names = ["pollen 0", "pollen 1"]

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

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# applying PCA
n_components = 195
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# train SVM classification
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("Best estimator found by grid search (SVC):")
print(clf.best_estimator_)

# evaluation of the model quality on the test set
y_pred = clf.predict(X_test_pca)
print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(2)))

# qualitative evaluation of the predictions using matplotlib
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

# plot the result of the prediction on a portion of the test set
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]
plot_gallery(X_test, prediction_titles, h, w)


# save result into a txt file
os.chdir('../..')
text_file = open("svc_result_new_samples_v4.txt", "a")
text_file.write("################################### n_components: " + repr(n_components))
text_file.write("\n" + classification_report(y_test, y_pred, target_names=target_names) + "\n")
text_file.close()

plt.show()