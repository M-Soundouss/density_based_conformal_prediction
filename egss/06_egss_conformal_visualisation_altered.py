# https://github.com/aqibsaeed/Occupancy-Detection/blob/master/Occupancy%20Detection.ipynb
# https://stackoverflow.com/questions/41577705/how-does-2d-kernel-density-estimation-in-python-sklearn-work
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from egss.egss_mlp_functions import kde2D_visualisation


# Read training and testing data
df_train = pd.read_json("EGSS_MLP_train_results.json")
df_outliers = pd.read_json("EGSS_MLP_outliers_results.json")
df_noisy = pd.read_json("EGSS_MLP_noisy_results.json")
df_clean = pd.read_json("EGSS_MLP_clean_results.json")

label_train = df_train["label"]
X_train = df_train["repr"]
label_outliers = df_outliers["label"]
label_noisy = df_noisy["label"]
X_outliers = df_outliers["repr"]
X_noisy = df_noisy["repr"]
label_clean = df_clean["label"]
X_clean = df_clean["repr"]

X_outliers = pd.DataFrame(X_outliers.tolist(), index=X_outliers.index)
X_noisy = pd.DataFrame(X_noisy.tolist(), index=X_noisy.index)
X_clean = pd.DataFrame(X_clean.tolist(), index=X_clean.index)
X_train = pd.DataFrame(X_train.tolist(), index=X_train.index)


X_train, X_val, label_train, label_val = train_test_split(X_train, label_train, test_size=0.1, random_state=1337,
                                                          stratify=label_train)

pca = PCA(n_components=2)

X_train = pca.fit_transform(X_train)
X_val = pca.transform(X_val)
X_clean = pca.transform(X_clean)
X_outliers = pca.transform(X_outliers)
X_noisy = pca.transform(X_noisy)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train-mean)/std
X_val = (X_val-mean)/std
X_clean = (X_clean-mean)/std
X_noisy = (X_noisy-mean)/std
X_outliers = (X_outliers-mean)/std

density_values = dict()
classes = [0, 1]

for i in classes:
    X_train_class = X_train[label_train == i, :]
    X_val_class = X_val[label_val == i, :]
    # alter next line command with X_clean/X_outliers/X_noisy and label_clean/label_outliers/label_noisy
    # (!!! keep X_noisy_class)
    X_noisy_class = X_noisy[label_noisy == i, :]

    feature_train_1 = X_train_class[:, 0]
    feature_train_2 = X_train_class[:, 1]

    feature_val_1 = X_val_class[:, 0]
    feature_val_2 = X_val_class[:, 1]

    feature_test_1 = X_noisy_class[:, 0]
    feature_test_2 = X_noisy_class[:, 1]

    xx, yy, zz, z_val = kde2D_visualisation(feature_train_1, feature_train_2, feature_val_1, feature_val_2, 0.1,
                                             min_=np.min(X_val) - 0.2, max_=np.max(X_val) + 0.2)

    xx, yy, zz, z_test = kde2D_visualisation(feature_train_1, feature_train_2, feature_test_1, feature_test_2, 0.1,
                                             min_=np.min(X_train) - 0.2, max_=np.max(X_train) + 0.2)
    scores = z_val.ravel().tolist()
    scores.sort()
    t = scores[(1 * len(scores) // 10)]
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

print(zz.shape)
plt.figure()
plt.rcParams.update({'font.size': 12})
plt.contourf(xx, yy, zz, alpha=0.4, levels=[-2, -1, 0, 1, 2], colors=('lightcoral', 'w', 'yellowgreen',
                                                                      'cornflowerblue'))
# subset_idx_0 = np.random.choice(range(len(density_values[0][3])), size=2000)
# subset_idx_1 = np.random.choice(range(len(density_values[1][3])), size=2000)

plt.scatter(density_values[0][3], density_values[0][4], c="black", s=0.3, alpha=0.4)
plt.scatter(density_values[1][3], density_values[1][4], c="black", s=0.3, alpha=0.4)

#plt.legend()
#plt.axis('off')

plt.xlim(-3, 20)
plt.ylim(-6, 6)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
# alter next line command with Original/Outlier/Noise in the title
title = 'EGSS Noise : ' + r'$\epsilon$' + ' = 0.1'
plt.title(title)
# alter next line command with clean/outlier/noise in the file name
plt.savefig('egss_density_noisy_eng.eps', format='eps')
plt.close()