# https://stackoverflow.com/questions/41577705/how-does-2d-kernel-density-estimation-in-python-sklearn-work
import numpy as np
import pandas as pd
import os, shutil
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# to do for each folder in clean/internet/masked/noisy/outlier
output_path = "c_images\\c_outlier_images"
df_test = pd.read_json("json\\celeba_outlier_cpred.json")

test_cpred = np.array(df_test["c_pred"].tolist())
test_pred = np.array(df_test["preds"].tolist())
test_img = np.array(df_test["image_id"].tolist())

cpreds_01 = list()
cpreds_null = list()

if os.path.exists(output_path):
    shutil.rmtree(output_path)

if not os.path.exists(output_path):
    os.makedirs(output_path)

size = 178, 218
for i in range(0, len(test_cpred)):
    try:
        test_img[i] = test_img[i].replace('\\', '/')
        input_shape = (178, 218)
        im = cv2.imread(test_img[i])
        im = cv2.resize(im, input_shape)
        cv2.imwrite(test_img[i], im)

        image_name = test_img[i].split("/")[-1]

        cv2.waitKey(0)

        text = "Pred:" + str((test_pred[i]))[0:4] + ",C_Pred:" + str(test_cpred[i])
        img = mpimg.imread(test_img[i])
        plt.imshow(img)
        plt.text(5, 5, text, bbox={'facecolor': 'white', 'pad': 10})
        plt.axis('off')
        plt.savefig(output_path + "/%s" % image_name)

        plt.close()
    except:
        print(i)