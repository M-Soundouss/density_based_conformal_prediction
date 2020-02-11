# https://github.com/aqibsaeed/Occupancy-Detection/blob/master/Occupancy%20Detection.ipynb
# https://stackoverflow.com/questions/41577705/how-does-2d-kernel-density-estimation-in-python-sklearn-work
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from celeba.celeba_cnn_functions import kde2D_visualisation


# Read training and testing data
df_train = pd.read_json("train_pred.json")
df_val = pd.read_json("val_pred.json")
df_test = pd.read_json("test_pred.json")

label_train = df_train["label"]
label_val = df_val["label"]
label_test = df_test["label"]
X_train = df_train["repr"]
X_val = df_val["repr"]
X_test = df_test["repr"]

X_train = pd.DataFrame(X_train.tolist(), index=X_train.index)
X_val = pd.DataFrame(X_val.tolist(), index=X_val.index)
X_test = pd.DataFrame(X_test.tolist(), index=X_test.index)

pca = PCA(n_components=2)

X_train = pca.fit_transform(X_train)
X_val = pca.transform(X_val)
X_test = pca.transform(X_test)

inertie = pca.explained_variance_ratio_
print(inertie)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train-mean)/std
X_val = (X_val-mean)/std
X_test = (X_test-mean)/std

density_values = dict()
classes = [0, 1]

for i in classes:
    X_train_class = X_train[label_train == i, :]
    X_val_class = X_val[label_val == i, :]
    X_test_class = X_test[label_test == i, :]

    feature_train_1 = X_train_class[:, 0]
    feature_train_2 = X_train_class[:, 1]

    feature_val_1 = X_val_class[:, 0]
    feature_val_2 = X_val_class[:, 1]

    feature_test_1 = X_test_class[:, 0]
    feature_test_2 = X_test_class[:, 1]

    xx, yy, zz, z_val = kde2D_visualisation(feature_train_1, feature_train_2, feature_val_1, feature_val_2, 0.1,
                                             min_=np.min(X_val) - 0.2, max_=np.max(X_val) + 0.2)

    xx, yy, zz, z_test = kde2D_visualisation(feature_train_1, feature_train_2, feature_test_1, feature_test_2, 0.1,
                                             min_=np.min(X_train) - 0.2, max_=np.max(X_train) + 0.2)
    scores = z_val.ravel().tolist()
    scores.sort()
    t = scores[(1 * len(scores) // 40)]
    print("t : ", t)
    plt.figure()
    zz_bis = np.where(zz > t, 1, 0)
    density_values[i] = [xx, yy, zz_bis, feature_test_1, feature_test_2]

zz_0, zz_1 = density_values[0][2], density_values[1][2]
xx, yy = density_values[0][0], density_values[0][1]

zz = zz_0.copy()

zz[zz_0>0] = -1
zz[zz_1>0] = 2
zz[np.logical_and(zz_0>0,zz_1>0)] = 1

plt.figure()
plt.rcParams.update({'font.size': 13})
plt.contourf(xx, yy, zz, alpha=0.4, levels=[-2, -1, 0, 1, 2], colors=('lightcoral', 'w', 'yellowgreen',
                                                                      'cornflowerblue'))

subset_idx_0 = np.random.choice(range(len(density_values[0][3])), size=2000)
subset_idx_1 = np.random.choice(range(len(density_values[1][3])), size=2000)

plt.scatter(density_values[0][3][subset_idx_0], density_values[0][4][subset_idx_0], c="r", s=0.3,
            label="Class 0 : Female")
plt.scatter(density_values[1][3][subset_idx_1], density_values[1][4][subset_idx_1], c="b", s=0.3,
            label="Class 1 : Male")
plt.legend()
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
#plt.axis('off')
title = 'CelebA : ' + r'$\epsilon$' + ' = 0.025'
plt.title(title)

plt.savefig('celeba_density_eng.eps', format='eps')
plt.close()