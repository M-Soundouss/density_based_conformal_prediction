# https://github.com/aqibsaeed/Occupancy-Detection/blob/master/Occupancy%20Detection.ipynb
# https://stackoverflow.com/questions/41577705/how-does-2d-kernel-density-estimation-in-python-sklearn-work
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from egss.egss_mlp_functions import prepare_data_visualisation, kde2D_visualisation

# Read training and testing data
df_train = pd.read_json("json\\egss_train_results.json")
df_test = pd.read_json("json\\egss_test_results.json")


X_train, label_train, X_val, label_val, X_test, label_test, inertie = prepare_data_visualisation(df_train, df_test)

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
plt.rcParams.update({'font.size': 13})
plt.contourf(xx, yy, zz, alpha=0.4, levels=[-2, -1, 0, 1, 2], colors=('lightcoral', 'w', 'yellowgreen',
                                                                      'cornflowerblue'))

subset_idx_0 = np.random.choice(range(len(density_values[0][3])), size=3000)
subset_idx_1 = np.random.choice(range(len(density_values[1][3])), size=3000)

plt.scatter(density_values[0][3][subset_idx_0], density_values[0][4][subset_idx_0], c="r", s=1.7,
            alpha=0.4, edgecolors="none", label="Class 0 : Unstable")
plt.scatter(density_values[1][3][subset_idx_1], density_values[1][4][subset_idx_1], c="b", s=1.7,
            alpha=0.4, edgecolors="none", label="Class 1 : Stable")
plt.legend()
plt.xlim(-3, 3)
plt.ylim(-3, 3)
#plt.axis('off')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
title = 'EGSS : ' + r'$\epsilon$' + ' = 0.1'
plt.title(title)

plt.savefig('egss_density_eng.eps', format='eps')
plt.close()

print(inertie)