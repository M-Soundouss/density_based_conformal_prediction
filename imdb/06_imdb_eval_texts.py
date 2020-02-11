# https://stackoverflow.com/questions/41577705/how-does-2d-kernel-density-estimation-in-python-sklearn-work
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imdb.imdb_nlp_functions import kde2D, kde2D_scores, kde2D_pred
import json


df_train = pd.read_json("IMDb_NLP_train_results.json")
df_clean = pd.read_json("IMDb_NLP_clean_results.json")
df_noisy = pd.read_json("IMDb_NLP_noisy_results.json")
df_outlier = pd.read_json("IMDb_NLP_outlier_results.json")

label_train = df_train["label"]
X_train = df_train["repr"]
label_clean = df_clean["label"]
label_noisy = df_noisy["label"]
label_outlier = df_outlier["label"]
X_clean = df_clean["repr"]
X_noisy = df_noisy["repr"]
X_outlier = df_outlier["repr"]

X_train, X_val, label_train, label_val = train_test_split(X_train, label_train, test_size=0.1, random_state=1337,
                                                      stratify=label_train)

label_train = np.array(label_train)
label_val = np.array(label_val)
label_noisy = np.array(label_noisy)
label_clean = np.array(label_clean)
label_outlier = np.array(label_outlier)

classes = [0, 1]
density_values = dict()

X_train_0 = np.array(X_train[label_train == 0].tolist())
X_val_0 = np.array(X_val[label_val == 0].tolist())

X_train_1 = np.array(X_train[label_train == 1].tolist())
X_val_1 = np.array(X_val[label_val == 1].tolist())

X_clean = np.array(X_clean.tolist())
X_noisy = np.array(X_noisy.tolist())
X_outlier = np.array(X_outlier.tolist())

model_0, z_val_0 = kde2D(X_train_0, X_val_0, 1)
print("training 0 complete")
scores_0 = z_val_0.ravel().tolist()
sorted_scores_0 = sorted(scores_0)

model_1, z_val_1 = kde2D(X_train_1, X_val_1, 1)
print("training 1 complete")
scores_1 = z_val_1.ravel().tolist()
sorted_scores_1 = sorted(scores_1)

scores_clean_00 = kde2D_scores(X_clean, model_0)
scores_clean_01 = kde2D_scores(X_clean, model_1)

scores_noisy_00 = kde2D_scores(X_noisy, model_0)
scores_noisy_01 = kde2D_scores(X_noisy, model_1)

scores_outlier_00 = kde2D_scores(X_outlier, model_0)
scores_outlier_01 = kde2D_scores(X_outlier, model_1)

print("Scores Calculation complete")

t_0 = sorted_scores_0[int(1 * len(sorted_scores_0)// 40)]
t_1 = sorted_scores_1[int(1 * len(sorted_scores_1)// 40)]

clean_pred = kde2D_pred(X_clean, t_0, t_1, scores_clean_00, scores_clean_01)
noisy_pred = kde2D_pred(X_noisy, t_0, t_1, scores_noisy_00, scores_noisy_01)
outlier_pred = kde2D_pred(X_noisy, t_0, t_1, scores_outlier_00, scores_outlier_01)
print("Prediction complete")

saved_elmts = list()

pred_clean = df_clean["preds"]
pred_noisy = df_noisy["preds"]
pred_outlier = df_outlier["preds"]

distance = abs(pred_noisy - pred_clean)

for i in range(len(distance)):
    if distance[i] >= (max(distance)/ 2):
        saved_elmts.append(i)

print(saved_elmts)

df_clean["c_pred"] = clean_pred.values()
df_clean = df_clean.drop("repr", axis=1)
df_clean.ix[saved_elmts].to_json('clean_cpred_imdb.json', orient='records')

df_noisy["c_pred"] = noisy_pred.values()
df_noisy = df_noisy.drop("repr", axis=1)
df_noisy.ix[saved_elmts].to_json('noisy_cpred_imdb.json', orient='records')

df_outlier["c_pred"] = outlier_pred.values()
df_outlier = df_outlier.drop("repr", axis=1)
df_outlier.ix[saved_elmts].to_json('outlier_cpred_imdb.json', orient='records')

file = "clean_cpred_imdb.json"
data = json.load(open(file, 'r'))
json.dump(data, open(file, 'w'), indent=4)

file = "outlier_cpred_imdb.json"
data = json.load(open(file, 'r'))
json.dump(data, open(file, 'w'), indent=4)

file = "noisy_cpred_imdb.json"
data = json.load(open(file, 'r'))
json.dump(data, open(file, 'w'), indent=4)