import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from celeba.celeba_cnn_functions import get_model, chunker


def read_and_resize(filepath):
    input_shape=(178, 218)
    im = cv2.imread(filepath)
    im = cv2.resize(im, input_shape)
    return np.array(im / (np.max(im)+ 0.001), dtype="float32")


if __name__ == "__main__":

    file_path = "gender.h5"

    # change output file name and folder path depending on the images you want to predict
    out_name = "outlier_images_pred.json"
    list_paths = glob("data\\outlier_images\\*.jpg")

    model, repr_model = get_model(n_classes=1)

    model.load_weights(file_path)
    repr_model.load_weights(file_path, by_name=True)

    preds = []
    repr_list = []
    image_paths = []

    for img_paths in tqdm(chunker(list_paths, 8)):
        images = np.array([read_and_resize(file_path) for file_path in img_paths])
        p, r = repr_model.predict(images)
        image_paths += img_paths
        repr_list += r.tolist()
        preds += p.ravel().tolist()

    image_paths = [x.split("/")[-1] for x in image_paths]

    df = pd.DataFrame({"image_id": image_paths, "preds":preds, "repr": repr_list})
    df.to_json(out_name, orient='records')

