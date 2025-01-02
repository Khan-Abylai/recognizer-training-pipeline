import glob
import shutil
import os
import random
import numpy as np
import cv2

with open("/mnt/sdc1/sng_eu/txt_detector/filenames_platesmania_sng.txt", "r") as f:
    files = f.read().splitlines()

print(len(files), files[0])
files = random.sample(files, 10000)
print(len(files), files[0])
out_img_path = "/mnt/sdc1/sng_eu/dataset/images"
out_label_path = "/mnt/sdc1/sng_eu/dataset/labels"

for idx, file in enumerate(files):
    img_path = file.replace("mnt_sda1", "mnt/sdc1")
    label_path = img_path.replace(".jpg", ".pb").replace(".jpeg", ".pb")
    if os.path.exists(img_path) and os.path.exists(label_path):
        label = np.loadtxt(label_path)
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        x1 = label[4]*w
        y1 = label[5]*h
        x2 = label[10]*w
        y2 = label[11]*h
        out_label = [0, x1, y1, x2, y2]
        new_label_path = os.path.join(out_label_path, os.path.basename(img_path).replace('.jpg', '.txt').replace('.jpeg', '.txt'))
        with open(new_label_path, "w") as f:
            f.writelines(' '.join(str(x) for x in out_label))
        shutil.copy(img_path, out_img_path)
        print(idx, img_path, x1, y1, x2, y2)


        # with open("label.txt", 'w') as f:
        #     f.writelines(' '.join(str(x) for x in out_label))
        #
        # cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), -1, cv2.LINE_AA )
        # cv2.imwrite("test.jpg", img)
        # break
