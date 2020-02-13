import cv2
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from celeba.celeba_cnn_functions import read_and_resize, chunker, gen, get_model, create_path
import os, shutil


if __name__ == "__main__":
    if os.path.exists("json"):
        shutil.rmtree("json")

    if not os.path.exists("json"):
        os.makedirs("json")

    base_path = "datasets\\CELEBA\\img_align_celeba\\"
    df_path = "datasets\\CELEBA\\list_attr_celeba.csv"

    data = pd.read_csv(df_path)

    print(data.shape)

    data = create_path(data, base_path=base_path)

    T_, test = train_test_split(data, test_size=0.1, random_state=1337)
    train, val = train_test_split(T_, test_size=0.1, random_state=1337)

    file_path = "gender.h5"

    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    early = EarlyStopping(monitor="val_acc", mode="max", patience=10)

    reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=3)

    callbacks_list = [checkpoint, early, reduce_on_plateau]  # early

    model, repr_model = get_model(n_classes=1)

    model.fit_generator(gen(train), validation_data=gen(val), epochs=20, verbose=2, workers=4,
                   callbacks=callbacks_list, steps_per_epoch=100, validation_steps=30, use_multiprocessing=True)

    model.load_weights(file_path)
    repr_model.load_weights(file_path, by_name=True)

    for df, out_name in zip([train, val, test], ["json\\celeba_train_pred.json", "json\\celeba_val_pred.json", "json\\celeba_test_pred.json"]):
        labels = []
        preds = []
        repr_list = []
        image_paths = []

        for img_paths, labs in tqdm(zip(chunker(df.path.tolist(), 128), chunker(df.Male.tolist(), 128))):
            labs = [int(g==1) for g in labs]
            images = np.array([read_and_resize(file_path) for file_path in img_paths])
            p, r = repr_model.predict(images)
            image_paths += img_paths
            repr_list += r.tolist()
            labels += labs
            preds += (p.ravel()>0.5).astype(int).tolist()

        print(out_name+" Accuracy : ", accuracy_score(labels, preds))
        image_paths = [x.split("/")[-1] for x in image_paths]

        df = pd.DataFrame({"image_id": image_paths, "preds":preds, "repr": repr_list, "label": labels})
        df.to_json(out_name, orient='records')