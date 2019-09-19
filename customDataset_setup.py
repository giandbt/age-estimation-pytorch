import pandas as pd
import os
import cv2
import json
import shutil
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def get_custom_csv(data_dir, data_type, output_dir, padding_perc = 0.4):

    images_dir = os.path.join(data_dir, 'images', data_type)
    annotations = os.path.join(data_dir, 'annotations', '%s_400_rigi.json' % data_type)

    with open(annotations, 'r') as f:
        annon_dict = json.loads(f.read())

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    os.makedirs(os.path.join(output_dir, 'images'))

    # Initializes variables
    avail_imgs = annon_dict.keys()
    image_paths = []
    age_list = []
    gender_list = []

    # Gets path for all images
    images = [os.path.join(images_dir, image) for image in os.listdir(images_dir) if 'jpg' or 'png' in image]

    for image in images:
        # read image (to determine size later)
        img = cv2.imread(image)

        # gets images Id
        img_id = os.path.basename(image)[:-4].lstrip('0')

        # ensures the image is in the dictionary key
        if not img_id in avail_imgs:
            continue

        for idx, annon in enumerate(annon_dict[img_id].keys()):

            # ensures we have a face detected
            if not annon_dict[img_id][annon]['age_gender_pred']:
                continue

            bbox = annon_dict[img_id][annon]['age_gender_pred']['detected']
            x1 = bbox['x1']
            x2 = bbox['x2']
            y1 = bbox['y1']
            y2 = bbox['y2']
            age = annon_dict[img_id][annon]['age_gender_pred']['predicted_ages']
            gender = annon_dict[img_id][annon]['age_gender_pred']['predicted_genders']

            # add padding to face
            upper_y = int(max(0, y1 - (y2 - y1) * padding_perc))
            lower_y = int(min(img.shape[0], y2 + (y2 - y1) * padding_perc))
            left_x = int(max(0, x1 - (x2 - x1) * padding_perc))
            right_x = int(min(img.shape[1], x2 + (x2 - x1) * padding_perc))
            face_im = img[upper_y: lower_y, left_x: right_x, ::-1]

            face_path = os.path.join(output_dir, 'images', '%s_%s_face.jpg' % (img_id, annon))

            Image.fromarray(np.uint8(face_im)).save(os.path.join(face_path))

            image_paths.append(face_path)
            age_list.append(age)
            gender_list.append(gender)

    # saves data in RetinaNet format
    data = {'Image_Path': image_paths, 'Age': age_list, 'Gender': gender_list}

    # Create DataFrame
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, '%s_labels_rigi.csv' % data_type), index=False, header=True)

def display_images(output_dir):
    csv_path = os.path.join(output_dir, 'train2017_labels.csv')
    df = pd.read_csv(str(csv_path))

    for _, row in df.iterrows():
        img_name = row["Image_Path"]
        age = row["Age"]
        img = cv2.imread(str(img_name), 1)
        cv2.imshow(str(round(age)), img)
        k = cv2.waitKey(0)
        if k == 27:
            break
        cv2.destroyAllWindows()

def get_custom_csv_filter(data_dir, data_type, output_dir, padding_perc = 0.4, threshold = 2500):

    images_dir = os.path.join(data_dir, 'images', data_type)
    annotations = os.path.join(data_dir, 'annotations', '%s_400.json' % data_type)

    with open(annotations, 'r') as f:
        annon_dict = json.loads(f.read())

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    os.makedirs(os.path.join(output_dir, 'images'))

    # Initializes variables
    avail_imgs = annon_dict.keys()
    image_paths = []
    age_list = []
    gender_list = []

    # Gets path for all images
    images = [os.path.join(images_dir, image) for image in os.listdir(images_dir) if 'jpg' or 'png' in image]

    for image in images:
        # read image (to determine size later)
        img = cv2.imread(image)

        # gets images Id
        img_id = os.path.basename(image)[:-4].lstrip('0')

        # ensures the image is in the dictionary key
        if not img_id in avail_imgs:
            continue

        for idx, annon in enumerate(annon_dict[img_id].keys()):

            # ensures we have a face detected
            if not annon_dict[img_id][annon]['age_gender_pred']:
                continue

            # ensures size is greater than threshold
            if annon_dict[img_id][annon]['face_area'] < threshold:
                continue

            bbox = annon_dict[img_id][annon]['age_gender_pred']['detected']
            x1 = bbox['x1']
            x2 = bbox['x2']
            y1 = bbox['y1']
            y2 = bbox['y2']
            age = annon_dict[img_id][annon]['age_gender_pred']['predicted_ages']
            gender = annon_dict[img_id][annon]['age_gender_pred']['predicted_genders']

            # add padding to face
            upper_y = int(max(0, y1 - (y2 - y1) * padding_perc))
            lower_y = int(min(img.shape[0], y2 + (y2 - y1) * padding_perc))
            left_x = int(max(0, x1 - (x2 - x1) * padding_perc))
            right_x = int(min(img.shape[1], x2 + (x2 - x1) * padding_perc))
            face_im = img[upper_y: lower_y, left_x: right_x, ::-1]

            face_path = os.path.join(output_dir, 'images', '%s_%s_face.jpg' % (img_id, annon))

            Image.fromarray(np.uint8(face_im)).save(os.path.join(face_path))

            image_paths.append(face_path)
            age_list.append(age)
            gender_list.append(gender)

    # saves data in RetinaNet format
    data = {'Image_Path': image_paths, 'Age': age_list, 'Gender': gender_list}

    # Create DataFrame
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, '%s_labels.csv' % data_type), index=False, header=True)

def get_custom_csv_labeled(output_dir):

    images_dir = os.path.join(output_dir, 'images')
    annotations = os.path.join(output_dir, 'annotations', 'filtered_annotations_15000_labeled.json')

    with open(annotations, 'r') as f:
        annon_dict = json.loads(f.read())

    # Initializes variables
    avail_imgs = annon_dict.keys()
    image_paths = []
    age_list = []
    gender_list = []

    # Gets path for all images
    images = [os.path.join(images_dir, image) for image in os.listdir(images_dir) if 'jpg' or 'png' in image]

    for image in images:
        # read image (to determine size later)
        img_id, annon, _ = os.path.basename(image)[:-4].split('_')

        age = annon_dict[img_id][annon]['age']
        gender = annon_dict[img_id][annon]['gender']

        face_path = os.path.join(output_dir, 'images', '%s_%s_face.jpg' % (img_id, annon))
        image_paths.append(face_path)
        age_list.append(age)
        gender_list.append(gender)

    # saves data in RetinaNet format
    data = {'Image_Path': image_paths, 'Age': age_list, 'Gender': gender_list}

    # Create DataFrame
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, 'train2017_labels_true.csv'), index=False, header=True)

def get_age_gender_charts(output_dir):

    save_dir = os.path.join(output_dir, 'distributions')
    # Gets path for all images
    images_dir = os.path.join(output_dir, 'images')
    images = [os.path.join(images_dir, image) for image in os.listdir(images_dir) if 'jpg' or 'png' in image]
    annotations = os.path.join(output_dir, 'annotations', 'train2017_400_rigi.json')

    with open(annotations, 'r') as file:
        results_dict = json.loads(file.read())

    ages = []
    genders = []

    for image in images:
        # read image (to determine size later)
        img_id, annon, _ = os.path.basename(image)[:-4].split('_')

        age = results_dict[img_id][annon]['age_gender_pred']['predicted_ages']
        gender = results_dict[img_id][annon]['age_gender_pred']['predicted_genders']
        ages.append(age)
        genders.append(gender)

    bins = range(0,100)
    counts, bins = np.histogram(ages, bins = bins)

    plt.figure(1)
    plt.hist(ages, bins=bins)
    plt.title("Age Distribution")

    # saving histogram
    plt.savefig(os.path.join(save_dir, 'age_histogram.jpg'), bbox_inches='tight')

    counts = [str(count) for count in counts]

    labels = ['%i to %i' % (bins[idx], bins[idx + 1]) for idx, bin in enumerate(bins[:-1])]
    labels[-1] = '%i and above' % bins[-2]

    # plotting pie chart
    plt.figure(2)
    plt.pie(counts, labels=labels, autopct='%1.1f%%')
    plt.title("Age Distribution")

    # saving pie chart
    plt.savefig(os.path.join(save_dir, 'ages_pie_chart.jpg'))
    
    man_counter = 0
    for gender in genders:
        if gender == 'man':
            man_counter +=1

    woman_counter = len(genders) - man_counter

    gender_counts = [man_counter,woman_counter]

    # plotting pie chart
    plt.figure(3)
    plt.pie(gender_counts, labels=['man', 'woman'], autopct='%1.1f%%')
    plt.title("Gender Distribution")

    # saving pie chart
    plt.savefig(os.path.join(save_dir, 'gender_pie_chart.jpg'))

def get_age_gender_charts_appa(output_dir):

    save_dir = os.path.join(output_dir, 'distributions')
    csv_path = os.path.join(output_dir, 'gt_avg_test.csv')
    df = pd.read_csv(str(csv_path))

    ages = []

    for _, row in df.iterrows():
        ages.append(row["apparent_age_avg"])

    bins = range(0,100)
    counts, bins = np.histogram(ages, bins = bins)

    plt.figure(1)
    plt.hist(ages, bins=bins)
    plt.title("Age Distribution")

    # saving histogram
    plt.savefig(os.path.join(save_dir, 'age_histogram.jpg'), bbox_inches='tight')

    counts = [str(count) for count in counts]

    labels = ['%i to %i' % (bins[idx], bins[idx + 1]) for idx, bin in enumerate(bins[:-1])]
    labels[-1] = '%i and above' % bins[-2]

    # plotting pie chart
    plt.figure(2)
    plt.pie(counts, labels=labels, autopct='%1.1f%%')
    plt.title("Age Distribution")

    # saving pie chart
    plt.savefig(os.path.join(save_dir, 'ages_pie_chart.jpg'))


if __name__ == '__main__':
    data_dir = '/home/giancarlo/Documents/custom_dataset_final_results/data'
    data_type = 'train2017'
    output_dir = '/home/giancarlo/Documents/Age-test/custom'
    #get_custom_csv(data_dir, data_type, output_dir)
    #display_images(output_dir)

    output_dir_2 = '/home/giancarlo/Documents/Age-test/custom_15000'
    #get_custom_csv_filter(data_dir, data_type, output_dir_2, padding_perc=0.4, threshold=15000)
    #display_images(output_dir_2)
    #get_custom_csv_labeled(output_dir_2)
    get_age_gender_charts(output_dir)

    output_dir_3 = '/home/giancarlo/Documents/Age-test/appa-real-release'
    #get_age_gender_charts_appa(output_dir_3)
