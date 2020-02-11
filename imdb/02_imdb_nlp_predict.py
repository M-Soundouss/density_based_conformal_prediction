# https://stackoverflow.com/questions/41577705/how-does-2d-kernel-density-estimation-in-python-sklearn-work
# https://medium.com/swlh/aps-failure-at-scania-trucks-203975cdc2dd

import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import pickle
from imdb.imdb_nlp_functions import get_model


if __name__ == "__main__":
    # import data
    import numpy as np
    data = pd.read_csv('datasets\\IMDB\\IMDB Dataset.csv', encoding='latin-1')
    data = data[['review', 'sentiment']]
    data = data.rename(columns={'sentiment': 'label', 'review': 'text'})

    # split data into train and test
    X = data['text']
    Y = data['label']
    Y = Y.replace("negative", 0)
    Y = Y.replace("positive", 1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1337, stratify=Y)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=1337,
                                                      stratify=Y_train)

    print(np.mean(Y))

    # process data (Tokenize, & Add padding)
    max_words = 7000
    max_len = 100

    # loading
    with open('imdb_tokenizer.pickle', 'rb') as handle:
        tok = pickle.load(handle)

    sequences = tok.texts_to_sequences(X_train)
    sequences_matrix_train = sequence.pad_sequences(sequences, maxlen=max_len)

    sequences = tok.texts_to_sequences(X_val)
    sequences_matrix_val = sequence.pad_sequences(sequences, maxlen=max_len)
    val = [sequences_matrix_val, Y_val]

    sequences = tok.texts_to_sequences(X_test)
    sequences_matrix_test = sequence.pad_sequences(sequences, maxlen=max_len)

    file_path = "imdb.h5"

    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    early = EarlyStopping(monitor="val_acc", mode="max", patience=10)

    reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=3)

    callbacks_list = [checkpoint, early, reduce_on_plateau]  # early

    model, repr_model = get_model(n_classes=1, lr=0.001, max_len=max_len, max_words=max_words)

    model.load_weights(file_path)
    repr_model.load_weights(file_path, by_name=True)

    pred = model.predict(sequences_matrix_test)
    pr, repr = repr_model.predict(sequences_matrix_test)

    mlp_preds = list()
    mlp_repr = list()
    for i in range(len(pred)):
        mlp_preds.append(pred[i][0])
        mlp_repr.append(repr[i])

    df = pd.DataFrame({"preds": mlp_preds, "repr": mlp_repr, "label": Y_test})
    df.to_json("IMDb_NLP_test_results.json", orient='records')

    pred_train = model.predict(sequences_matrix_train)
    pr, repr_train = repr_model.predict(sequences_matrix_train)

    mlp_preds = list()
    mlp_repr = list()
    for i in range(len(pred_train)):
        mlp_preds.append(pred_train[i][0])
        mlp_repr.append(repr_train[i])

    df = pd.DataFrame({"preds": mlp_preds, "repr": mlp_repr, "label": Y_train})
    df.to_json("IMDb_NLP_train_results.json", orient='records')

    pred_val = model.predict(sequences_matrix_val)
    pr, repr_val = repr_model.predict(sequences_matrix_val)

    mlp_preds = list()
    mlp_repr = list()
    for i in range(len(pred_val)):
        mlp_preds.append(pred_val[i][0])
        mlp_repr.append(repr_val[i])

    df = pd.DataFrame({"preds": mlp_preds, "repr": mlp_repr, "label": Y_val})
    df.to_json("IMDb_NLP_val_results.json", orient='records')

    import json

    file = "IMDb_NLP_test_results.json"
    data = json.load(open(file, 'r'))
    json.dump(data, open(file, 'w'), indent=4)

    file = "IMDb_NLP_train_results.json"
    data = json.load(open(file, 'r'))
    json.dump(data, open(file, 'w'), indent=4)

    file = "IMDb_NLP_val_results.json"
    data = json.load(open(file, 'r'))
    json.dump(data, open(file, 'w'), indent=4)