import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from imdb.imdb_nlp_functions import get_model
import pickle
import numpy as np

if __name__ == "__main__":
    # import data
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

    # process data (Tokenize, & Add padding)
    max_words = 7000
    max_len = 100
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(X_train)
    print(len(tok.word_counts))

    sequences = tok.texts_to_sequences(X_train)
    print(np.quantile([len(x) for x in sequences], 0.95))
    sequences_matrix_train = sequence.pad_sequences(sequences, maxlen=max_len)

    sequences = tok.texts_to_sequences(X_val)
    sequences_matrix_val = sequence.pad_sequences(sequences, maxlen=max_len)
    val = [sequences_matrix_val, Y_val]

    # saving tokenizer
    with open('imdb_tokenizer.pickle', 'wb') as handle:
        pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)

    file_path = "imdb.h5"

    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    early = EarlyStopping(monitor="val_acc", mode="max", patience=10)

    reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=3)

    callbacks_list = [checkpoint, early, reduce_on_plateau]  # early

    model, repr_model = get_model(n_classes=1, lr=0.001, max_len=max_len, max_words=max_words)

    model.fit(x=sequences_matrix_train, y=Y_train, validation_data=val, epochs=5, verbose=2, callbacks=callbacks_list)