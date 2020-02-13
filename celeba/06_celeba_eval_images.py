# https://stackoverflow.com/questions/41577705/how-does-2d-kernel-density-estimation-in-python-sklearn-work
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import os, shutil


def kde2D(X_train, X_val, bandwidth, **kwargs):
    """Build 2D kernel density estimate (KDE)."""

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(X_train)

    # score_samples() returns the log-likelihood of the samples
    z_val = kde_skl.score_samples(X_val)
    return kde_skl, z_val.ravel()


def kde2D_pred(X, density_values):
    """Predictions based on conformal prediction and 2D kernel density estimate (KDE)."""
    preds = dict()

    # score_samples() returns the log-likelihood of the samples
    scores_0 = density_values[0][0].score_samples(X).ravel().tolist()
    scores_1 = density_values[1][0].score_samples(X).ravel().tolist()


    for i in range(0, len(X)):
        preds[i] = list()
        if scores_0[i] >= density_values[0][1]:
            preds[i].append(0)
        if scores_1[i] >= density_values[1][1]:
            preds[i].append(1)
    return preds


df_train = pd.read_json("json\\celeba_train_pred.json")
df_val = pd.read_json("json\\celeba_val_pred.json")
df_test = pd.read_json("json\\celeba_test_pred.json")
df_clean = pd.read_json("json\\celeba_clean_images_pred.json")
df_noisy = pd.read_json("json\\celeba_noisy_images_pred.json")
df_internet = pd.read_json("json\\celeba_internet_images_pred.json")
df_outlier = pd.read_json("json\\celeba_outlier_images_pred.json")
df_masked = pd.read_json("json\\celeba_masked_images_pred.json")

classes = [0, 1]
density_values = dict()

for i in classes:
    df_train_i = df_train[df_train.label == i]
    train_X = np.array(df_train_i["repr"].tolist())

    df_val_i = df_val[df_val.label == i]
    val_X = np.array(df_val_i["repr"].tolist())

    model, z_val = kde2D(train_X, val_X, 1)
    scores = z_val.ravel().tolist()
    sorted_scores = sorted(scores)
    t = sorted_scores[(1 * len(sorted_scores) // 40)]

    density_values[i] = [model, t]

test_X = np.array(df_test["repr"].tolist())
test_pred = kde2D_pred(test_X, density_values)

clean_X = np.array(df_clean["repr"].tolist())
clean_pred = kde2D_pred(clean_X, density_values)

noisy_X = np.array(df_noisy["repr"].tolist())
noisy_pred = kde2D_pred(noisy_X, density_values)

masked_X = np.array(df_masked["repr"].tolist())
masked_pred = kde2D_pred(masked_X, density_values)

outlier_X = np.array(df_outlier["repr"].tolist())
outlier_pred = kde2D_pred(outlier_X, density_values)

internet_X = np.array(df_internet["repr"].tolist())
internet_pred = kde2D_pred(internet_X, density_values)

df_clean["c_pred"] = clean_pred.values()
df_clean.to_json('json\\celeba_clean_cpred.json', orient='records')

df_noisy["c_pred"] = noisy_pred.values()
df_noisy.to_json('json\\celeba_noisy_cpred.json', orient='records')

df_masked["c_pred"] = masked_pred.values()
df_masked.to_json('json\\celeba_masked_cpred.json', orient='records')

df_outlier["c_pred"] = outlier_pred.values()
df_outlier.to_json('json\\celeba_outlier_cpred.json', orient='records')

df_internet["c_pred"] = internet_pred.values()
df_internet.to_json('json\\celeba_internet_cpred.json', orient='records')

if os.path.exists("c_images"):
    shutil.rmtree("c_images")

if not os.path.exists("c_images"):
    os.makedirs("c_images")
