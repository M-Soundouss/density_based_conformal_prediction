# https://github.com/aqibsaeed/Occupancy-Detection/blob/master/Occupancy%20Detection.ipynb
# https://stackoverflow.com/questions/41577705/how-does-2d-kernel-density-estimation-in-python-sklearn-work
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from imdb.imdb_nlp_functions import cp_plots, calculate_accuracy, kde2D, kde2D_scores, kde2D_pred


# Read training and testing data
df_train = pd.read_json("json\\imdb_train_results.json")
df_test = pd.read_json("json\\imdb_test_results.json")
df_val = pd.read_json("json\\imdb_val_results.json")

df_train["preds"] = df_train["preds"].apply(lambda x: 0 if x <= 0.5 else 1)
df_test["preds"] = df_test["preds"].apply(lambda x: 0 if x <= 0.5 else 1)
df_val["preds"] = df_val["preds"].apply(lambda x: 0 if x <= 0.5 else 1)

# select X & Y for each dataset
classes = [0, 1]

alphas_results = dict()

alphas = [0.01*i for i in range(51)]

train_repr = df_train["repr"]
train_mlp_pred = df_train["preds"]
train_labels = df_train["label"]

val_repr = df_val["repr"]
val_mlp_pred = df_val["preds"]
val_labels = df_val["label"]

test_repr = df_test["repr"]
test_mlp_pred = df_test["preds"]
test_labels = df_test["label"]

X_train_0 = np.array(train_repr[train_labels == 0].tolist())
X_val_0 = np.array(val_repr[val_labels == 0].tolist())

X_train_1 = np.array(train_repr[train_labels == 1].tolist())
X_val_1 = np.array(val_repr[val_labels == 1].tolist())

X_test = np.array(test_repr.tolist())

model_0, z_val_0 = kde2D(X_train_0, X_val_0, 1)
print("training 0 complete")
scores_0 = z_val_0.ravel().tolist()
sorted_scores_0 = sorted(scores_0)

model_1, z_val_1 = kde2D(X_train_1, X_val_1, 1)
print("training 1 complete")
scores_1 = z_val_1.ravel().tolist()
sorted_scores_1 = sorted(scores_1)

scores_00 = kde2D_scores(X_test, model_0)
scores_01 = kde2D_scores(X_test, model_1)

print("Scores Calculation complete")

cnn_acc = accuracy_score(test_labels,test_mlp_pred)

for alpha in alphas:
    print(alpha)
    t_0 = sorted_scores_0[int(alpha * len(sorted_scores_0))]
    t_1 = sorted_scores_1[int(alpha * len(sorted_scores_1))]

    test_pred = kde2D_pred(X_test, t_0, t_1, scores_00, scores_01)
    print("Prediction complete")

    valid_cp_acc, valid_cnn_acc, cp_01_acc, percentage_null, percentage_01 = \
        calculate_accuracy(df_test, test_labels, test_mlp_pred, test_pred)

    alphas_results[alpha] = [cnn_acc, valid_cp_acc, valid_cnn_acc,
                             cp_01_acc, percentage_null, percentage_01]
    print("Calculations done")

alphas_results_df = pd.DataFrame.from_dict(
    data=alphas_results, orient='index',
    columns=["CNN Accuracy", "Valid Conformal Prediction Accuracy", "Valid CNN Accuracy",
             "Conformal Accuracy With {0,1}", "Null Percentage", "{0,1} Percentage"])
print(alphas_results_df)
alphas_results_df.to_json('json\\imdb_alphas_results.json')
df_results = pd.read_json("json\\imdb_alphas_results.json", convert_axes=False)

cp_plots(df_results, "IMDb", "GRU")