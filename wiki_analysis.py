import scipy.io
import datetime
import pandas as pd
import os
import numpy as np
import cv2

def get_wiki_csv(data_dir, padding_perc = 0.4):

    mat = scipy.io.loadmat(os.path.join(data_dir, 'wiki.mat'))

    # Get Image path
    image_paths = mat['wiki'][0][0][2][0]
    image_paths = [image_path[0] for image_path in image_paths]

    # Get Age
    date_of_births = mat['wiki'][0][0][0][0]
    date_of_births = [datetime.datetime.fromordinal(int(date_of_birth)).year -1 for date_of_birth in date_of_births]

    year_taken = mat['wiki'][0][0][1][0]

    age = year_taken - date_of_births

    # Get Gender
    gender = mat['wiki'][0][0][3][0]

    # Get Face
    faces = mat['wiki'][0][0][5][0]
    faces = [face[0] for face in faces]

    ignored = []
    face_paths = []

    for idx, image in enumerate(image_paths):
        image_path = os.path.join(data_dir, image)
        face_paths.append(os.path.join(data_dir, image[:-4] + '_face.jpg'))

        img = cv2.imread(image_path, 1)

        if img is None:
            ignored.append(image_path[:-4] + '_face.jpg')
            continue

        face = faces[idx]

        y1 = int(face[0])
        y2 = int(face[2])
        x1 = int(face[1])
        x2 = int(face[3])

        upper_y = int(max(0, face[1] - (y2 - y1) * padding_perc))
        lower_y = int(min(img.shape[0], face[3] + (y2 - y1) * padding_perc))
        left_x = int(max(0, face[0] - (x2 - x1) * padding_perc))
        right_x = int(min(img.shape[1], face[2] + (x2 - x1) * padding_perc))
        face_im = img[upper_y: lower_y, left_x: right_x]

        if face_im.shape[0] == 0 or face_im.shape[1] == 0:
            ignored.append(image_path[:-4] + '_face.jpg')
            continue

        cv2.imwrite(image_path[:-4] + '_face.jpg', face_im)


    # Save CSV for training and validation
    data = {'Image_Path': face_paths, 'Age': age, 'Gender': gender, 'Face': faces}
    df = pd.DataFrame(data)

    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    validation = df[~msk]

    train.to_csv('wiki_train.csv', index=False)

    validation.to_csv('wiki_val.csv', index=False)

    # Save ignore list
    ignore_dict = {"Image_Path": ignored}
    df_ignored = pd.DataFrame(ignore_dict)
    df_ignored.to_csv('ignore.csv', index=False)


if __name__ == '__main__':
    data_dir = '/home/giancarlo/Documents/Age-test/wiki'
    get_wiki_csv(data_dir)