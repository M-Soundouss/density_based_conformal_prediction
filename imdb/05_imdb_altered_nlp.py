import os
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import pickle
from imdb.imdb_nlp_functions import get_model


def random_replace(sentence, replace_proba=0.3):
    return " ".join([a if np.random.random() > replace_proba else np.random.choice(list(vocab.keys()))
                     for a in sentence.split()])


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

    # process data (Tokenize, & Add padding)
    max_words = 7000
    max_len = 100

    # loading
    with open('imdb_tokenizer.pickle', 'rb') as handle:
        tok = pickle.load(handle)

    vocab = tok.word_index

    X_test = X_test.tolist()[:100]
    X_noise = list()
    X_outlier = list()
    Y_test = Y_test[:100]

    for i in range(len(X_test)):
        X_noise.append(random_replace(X_test[i]))
        X_outlier.append(random_replace(X_test[i], 1))

    sequences = tok.texts_to_sequences(X_test)
    sequences_matrix_test = sequence.pad_sequences(sequences, maxlen=max_len)

    sequences = tok.texts_to_sequences(X_noise)
    sequences_matrix_noise = sequence.pad_sequences(sequences, maxlen=max_len)

    sequences = tok.texts_to_sequences(X_outlier)
    sequences_matrix_outlier = sequence.pad_sequences(sequences, maxlen=max_len)

    file_path = "imdb.h5"

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

    df = pd.DataFrame({"data": X_test, "preds": mlp_preds, "repr": mlp_repr, "label": Y_test})
    df.to_json("IMDb_NLP_clean_results.json", orient='records')

    pred = model.predict(sequences_matrix_noise)
    pr, repr = repr_model.predict(sequences_matrix_noise)

    mlp_preds = list()
    mlp_repr = list()
    for i in range(len(pred)):
        mlp_preds.append(pred[i][0])
        mlp_repr.append(repr[i])

    df = pd.DataFrame({"data": X_noise, "preds": mlp_preds, "repr": mlp_repr, "label": Y_test})
    df.to_json("IMDb_NLP_noisy_results.json", orient='records')

    pred = model.predict(sequences_matrix_outlier)
    pr, repr = repr_model.predict(sequences_matrix_outlier)

    mlp_preds = list()
    mlp_repr = list()
    for i in range(len(pred)):
        mlp_preds.append(pred[i][0])
        mlp_repr.append(repr[i])

    df = pd.DataFrame({"data": X_outlier, "preds": mlp_preds, "repr": mlp_repr, "label": Y_test})
    df.to_json("IMDb_NLP_outlier_results.json", orient='records')