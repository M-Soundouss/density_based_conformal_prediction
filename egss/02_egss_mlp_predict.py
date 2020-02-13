# https://stackoverflow.com/questions/41577705/how-does-2d-kernel-density-estimation-in-python-sklearn-work
# https://medium.com/swlh/aps-failure-at-scania-trucks-203975cdc2dd
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from egss.egss_mlp_functions import  get_model
import os, shutil


if __name__ == "__main__":
    output_path = "json"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    data = pd.read_csv("datasets\\EGSS\\Data_for_UCI_named.csv", header=0)

    data = data.replace("unstable", 0)
    data = data.replace("stable", 1)
    data = data.drop("stab", axis=1)

    Y = data["stabf"]
    X = data.drop("stabf", axis=1)
    X = (X - X.mean()) / X.std()

    print(X.shape)

    train_data, test_data , train_label, test_label = train_test_split(X, Y, test_size=0.2,
                                                                     random_state=1337, stratify = Y)

    train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=0.1,
                                                                      random_state=1337, stratify=train_label)

    file_path = "egss.h5"

    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    early = EarlyStopping(monitor="val_acc", mode="max", patience=10)

    reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=3)

    callbacks_list = [checkpoint, early, reduce_on_plateau]  # early

    model, repr_model = get_model(n_classes=1)

    model.load_weights(file_path)
    repr_model.load_weights(file_path, by_name=True)

    # train data
    pred_train = model.predict(train_data)
    pr, repr_train = repr_model.predict(train_data)

    mlp_preds = list()
    mlp_repr = list()
    for i in range(len(pred_train)):
        mlp_preds.append(pred_train[i][0])
        mlp_repr.append(repr_train[i])

    df = pd.DataFrame({"preds": mlp_preds, "repr": mlp_repr, "label": train_label})
    df.to_json(output_path + "\\egss_train_results.json", orient='records')

    pred = model.predict(test_data)
    pr, repr = repr_model.predict(test_data)

    mlp_preds = list()
    mlp_repr = list()
    for i in range(len(pred)):
        mlp_preds.append(pred[i][0])
        mlp_repr.append(repr[i])

    df = pd.DataFrame({"preds": mlp_preds, "repr": mlp_repr, "label": test_label})
    df.to_json(output_path + "\\egss_test_results.json", orient='records')

    pred = model.predict(val_data)
    pr, repr = repr_model.predict(val_data)

    mlp_preds = list()
    mlp_repr = list()
    for i in range(len(pred)):
        mlp_preds.append(pred[i][0])
        mlp_repr.append(repr[i])

    df = pd.DataFrame({"preds": mlp_preds, "repr": mlp_repr, "label": val_label})
    df.to_json(output_path + "\\egss_val_results.json", orient='records')



