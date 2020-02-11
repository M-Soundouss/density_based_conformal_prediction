import cv2
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Dropout, GlobalMaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity


# define helper functions
def kde2D_visualisation(x1, y1, x2, y2, bandwidth, xbins=100j, ybins=100j, min_=0, max_=10, **kwargs):
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[min_:max_:xbins,
             min_:max_:ybins]
    print(xx.shape)

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train = np.vstack([y1, x1]).T
    xy_test = np.vstack([y2, x2]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z_grid = np.exp(kde_skl.score_samples(xy_sample))
    z_test = np.exp(kde_skl.score_samples(xy_test))
    return xx, yy, np.reshape(z_grid, xx.shape), z_test.ravel()


def kde2D(X_train, X_val, bandwidth, **kwargs):
    """Build 2D kernel density estimate (KDE)."""

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(X_train)

    # score_samples() returns the log-likelihood of the samples
    z_val = kde_skl.score_samples(X_val)
    return kde_skl, z_val.ravel()


def kde2D_scores(X, model):
    """Scores based on conformal prediction and 2D kernel density estimate (KDE)."""
    # score_samples() returns the log-likelihood of the samples
    scores = model.score_samples(X).ravel().tolist()

    return scores


def kde2D_pred(X, t_0, t_1, scores_0, scores_1):
    """Predictions based on conformal prediction and 2D kernel density estimate (KDE)."""
    preds = dict()

    # score_samples() returns the log-likelihood of the samples

    for i in range(0, len(X)):
        preds[i] = list()
        if scores_0[i] >= t_0:
            preds[i].append(0)
        if scores_1[i] >= t_1:
            preds[i].append(1)
    return preds


def cp_plots(df_results, name, model):
    texte = model + " Model"
    title_axis = r'$\epsilon$' + ' Values'
    title_acc = 'Accuracy per ' + r'$\epsilon$' + ' Values for ' + name
    title_per = 'Percentages per ' + r'$\epsilon$' + ' Values for ' + name

    df_percentages = df_results[
        ["{0,1} Percentage", "Null Percentage", "Conformal Accuracy With {0,1}"]].copy().sort_index()

    axis = df_percentages.index.values

    plt.rcParams.update({'font.size': 13})
    plt.plot(axis, df_percentages[["{0,1} Percentage"]], color='#2b83ba',
             linestyle='-', linewidth=2, label="{0,1} Set")
    plt.plot(axis, df_percentages[["Null Percentage"]], color='#abdda4',
             linestyle='--', linewidth=2, label="Empty Set")
    plt.legend()
    plt.xlabel(title_axis)
    plt.ylabel("Percentage")

    plt.title(title_per)
    plt.legend(loc="upper right")
    plt.xticks(ticks=[10 * i for i in range(6)])

    plt.savefig('Percentages_eng_%s.eps'%name, format='eps')
    plt.show()

    df_accuracies = df_results[["Valid Conformal Prediction Accuracy",
                                "Valid CNN Accuracy", "CNN Accuracy"]].copy().sort_index()

    axis = df_accuracies.index.values

    plt.rcParams.update({'font.size': 13})
    plt.plot(axis, df_accuracies[["Valid Conformal Prediction Accuracy"]], color='#d7191c',
             linestyle='-', linewidth=2, label="Valid Conformal Model")
    plt.plot(axis, df_accuracies[["Valid CNN Accuracy"]], color='#fdae61',
             linestyle='--', linewidth=2, label= "Valid " + texte)
    plt.plot(axis, df_accuracies[["CNN Accuracy"]], color='#2b83ba',
             linestyle='-.', linewidth=2, label=texte)
    plt.legend()
    plt.xlabel(title_axis)
    plt.ylabel("Accuracy")

    plt.title(title_acc)
    plt.legend(loc="bottom left")
    plt.xticks(ticks=[10 * i for i in range(6)])

    plt.savefig('Accuracies_eng_%s.eps'%name, format='eps')
    plt.show()


def calculate_accuracy(df_test, test_labels, test_mlp_pred, test_pred):
    nbr_all = df_test.shape[0]
    valid_cp = []
    nbr_valid_cnn = 0
    nbr_cp_01 = 0
    nbr_null = 0
    nbr_01 = 0
    labels = test_labels
    df_test = pd.DataFrame()
    df_test["c_pred"] = test_pred.values()
    df_test.index = labels.index
    c_preds = df_test["c_pred"]

    for i in labels.index:
        if labels[i] in c_preds[i]:
            nbr_cp_01 += 1
            if len(c_preds[i]) == 1:
                valid_cp.append(1)
                if test_mlp_pred[i] == labels[i]:
                    nbr_valid_cnn += 1
            else:
                if len(c_preds[i]) > 1:
                    nbr_01 += 1
        else:
            if len(c_preds[i]) == 0:
                nbr_null += 1
            if len(c_preds[i]) == 1:
                valid_cp.append(0)
                if test_mlp_pred[i] == labels[i]:
                    nbr_valid_cnn += 1

    cp_01_acc = nbr_cp_01 / nbr_all
    valid_cp_acc = np.mean(valid_cp)
    valid_cnn_acc = nbr_valid_cnn / len(valid_cp)
    percentage_null = nbr_null / nbr_all * 100
    percentage_01 = nbr_01 / nbr_all * 100

    return (valid_cp_acc, valid_cnn_acc, cp_01_acc, percentage_null, percentage_01)


def prepare_data_visualisation(df_train, df_test):
    label_train = df_train["label"]
    label_test = df_test["label"]
    X_train = df_train["repr"]
    X_test = df_test["repr"]

    X_train = pd.DataFrame(X_train.tolist(), index=X_train.index)
    X_test = pd.DataFrame(X_test.tolist(), index=X_test.index)

    X_train, X_val, label_train, label_val = train_test_split(X_train, label_train, test_size=0.1, random_state=1337,
                                                          stratify=label_train)

    pca = PCA(n_components=2)

    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    X_train = (X_train-mean)/std
    X_val = (X_val-mean)/std
    X_test = (X_test-mean)/std

    inertie = pca.explained_variance_ratio_
    return(X_train, label_train, X_val, label_val, X_test, label_test, inertie)


def read_and_resize(filepath):
    im = cv2.imread(filepath)

    return np.array(im / (np.max(im)+ 0.001), dtype="float32")


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def gen(df, batch_size=32):
    df = df.sample(frac=1)
    while True:
        for i, batch in enumerate([df[i:i+batch_size] for i in range(0,df.shape[0],batch_size)]):
            images = np.array([read_and_resize(file_path) for file_path in batch.path.values])

            labels = np.array([int(g==1) for g in batch.Male.values])

            yield images, labels


def get_model(n_classes=1, lr=0.00001):

    base_model = ResNet50(weights='imagenet', include_top=False)

    x = base_model.output
    x = Dropout(0.1)(x)
    x = GlobalMaxPooling2D()(x)
    x = Dropout(0.1)(x)
    x_repr = Dense(50, activation="tanh", name="repr_l")(x)
    x = Dropout(0.1)(x_repr)

    if n_classes == 1:
        x = Dense(n_classes, activation="sigmoid", name='out')(x)
    else:
        x = Dense(n_classes, activation="softmax", name='out')(x)

    base_model = Model(base_model.input, x, name="base_model")
    repr_model = Model(base_model.input, [x, x_repr], name="repr_model")
    if n_classes == 1:
        base_model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(lr))
        repr_model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(lr))

    else:
        base_model.compile(loss="sparse_categorical_crossentropy", metrics=['acc'], optimizer=Adam(lr))
        repr_model.compile(loss="sparse_categorical_crossentropy", metrics=['acc'], optimizer=Adam(lr))

    return base_model, repr_model


def create_path(df, base_path):

    df['path'] = df.apply(lambda x: base_path+x['image_id'], axis=1)

    return df