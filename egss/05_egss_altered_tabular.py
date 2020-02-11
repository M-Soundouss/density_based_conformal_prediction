import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from egss.egss_mlp_functions import get_model


def random_replace(array, replace_proba=0.2, range=5):
    return [a if np.random.random() > replace_proba else np.random.uniform(-range, range) for a in array]

data = pd.read_csv("datasets\\EGSS\\Data_for_UCI_named.csv", header=0)
data_outliers = pd.read_csv("datasets\\EGSS\\Data_for_UCI_named.csv", header=0)

data = data.replace("unstable", 0)
data = data.replace("stable", 1)
data = data.drop("stab", axis=1)

Y = data["stabf"]
X = data.drop("stabf", axis=1)


data_outliers = data_outliers.replace("unstable", 0)
data_outliers = data_outliers.replace("stable", 1)
data_outliers = data_outliers.drop("stab", axis=1)

Y_outliers = data_outliers["stabf"]
X_outliers = data_outliers.drop("stabf", axis=1)

X_outliers = X_outliers.values.tolist()
_X_outliers = [random_replace(a) for a in X_outliers]
distances = [np.linalg.norm(np.array(a)- np.array(b)) for a, b in zip(X_outliers, _X_outliers)]
X_Y_distance = list(zip(_X_outliers, Y_outliers, distances))
X_Y_distance.sort(key=lambda x: x[2], reverse=True)
X_Y_distance = X_Y_distance[:len(X_Y_distance)//5]

X_outliers = [a[0] for a in X_Y_distance]
Y_outliers = [a[1] for a in X_Y_distance]

X_outliers = np.array(X_outliers)

X_outliers = pd.DataFrame(X_outliers, columns=X.columns)



train_data, test_data, train_label, test_label = train_test_split(X, Y, test_size=0.2,
                                                                  random_state=1337, stratify=Y)
_, X_outliers, _, Y_outliers = train_test_split(X_outliers, Y_outliers, test_size=0.2,
                                                                  random_state=1337, stratify=Y_outliers)

file_path = "egss.h5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=10)
reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=3)
callbacks_list = [checkpoint, early, reduce_on_plateau]  # early
model, repr_model = get_model(n_classes=1)
model.load_weights(file_path)
repr_model.load_weights(file_path, by_name=True)

# noisy data
mu, sigma = 0, 0.5
# creating a noise with the same dimension as the dataset
noise = np.random.normal(mu, sigma, test_data.shape)
noisy_test_data = test_data + noise

noisy_test_data = (noisy_test_data - X.mean()) / X.std()

pred_noisy = model.predict(noisy_test_data)
pr, repr_noisy = repr_model.predict(noisy_test_data)

mlp_preds_noisy = list()
mlp_repr_noisy = list()
for i in range(len(pred_noisy)):
    mlp_preds_noisy.append(pred_noisy[i][0])
    mlp_repr_noisy.append(repr_noisy[i])

df = pd.DataFrame({"preds": mlp_preds_noisy, "repr": mlp_repr_noisy, "label": test_label})
df.to_json("EGSS_MLP_noisy_results.json", orient='records')


X_outliers = (X_outliers - X.mean()) / X.std()

pred_noisy = model.predict(X_outliers)
pr, repr_noisy = repr_model.predict(X_outliers)

mlp_preds_noisy = list()
mlp_repr_noisy = list()
for i in range(len(pred_noisy)):
    mlp_preds_noisy.append(pred_noisy[i][0])
    mlp_repr_noisy.append(repr_noisy[i])

df = pd.DataFrame({"preds": mlp_preds_noisy, "repr": mlp_repr_noisy, "label": Y_outliers})
df.to_json("EGSS_MLP_outliers_results.json", orient='records')

test_data = (test_data - X.mean()) / X.std()

pred_noisy = model.predict(test_data)
pr, repr_noisy = repr_model.predict(test_data)

mlp_preds_noisy = list()
mlp_repr_noisy = list()
for i in range(len(pred_noisy)):
    mlp_preds_noisy.append(pred_noisy[i][0])
    mlp_repr_noisy.append(repr_noisy[i])

df = pd.DataFrame({"preds": mlp_preds_noisy, "repr": mlp_repr_noisy, "label": test_label})
df.to_json("EGSS_MLP_clean_results.json", orient='records')

