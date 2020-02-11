# https://stackoverflow.com/questions/41577705/how-does-2d-kernel-density-estimation-in-python-sklearn-work
# https://medium.com/swlh/aps-failure-at-scania-trucks-203975cdc2dd
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from egss.egss_mlp_functions import get_model


if __name__ == "__main__":

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

    val = [val_data, val_label]

    file_path = "egss.h5"

    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    early = EarlyStopping(monitor="val_acc", mode="max", patience=10)

    reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=3)

    callbacks_list = [checkpoint, early, reduce_on_plateau]  # early

    model, repr_model = get_model(n_classes=1, lr=0.01)

    model.fit(x=train_data, y=train_label, validation_data=val, epochs=15, verbose=2, callbacks=callbacks_list)