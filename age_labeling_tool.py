import os
import cv2
import json

def age_labeling(output_dir, image_idx = 0):

    # get path for all faces
    images = [os.path.join(output_dir, 'images', image) for image in os.listdir(os.path.join(output_dir, 'images')) if 'jpg' in image]

    # reads original json file if labeling has not started. Otherwise, open the json file that already modify some labels.
    if image_idx == 0:
        with open(os.path.join(output_dir, 'annotations', 'filtered_annotations_15000.json'), 'r') as f:
            annon_dict = json.loads(f.read())
    else:
        with open(os.path.join(output_dir, 'annotations', 'filtered_annotations_15000_labeled.json'), 'r') as f:
            annon_dict = json.loads(f.read())

    num_faces = len(images)

    while image_idx < (num_faces-2):
        print('Analyzing Face %i out of %i' % (image_idx + 1, num_faces))
        image = images[image_idx]
        img_id, annon, _ = os.path.basename(image)[:-4].split('_')
        print(img_id)

        if annon_dict[img_id][annon]['age'] == {}:
            image_idx += 1
            continue

        label = annon_dict[img_id][annon]['age']

        img = cv2.imread(image, 1)
        cv2.destroyAllWindows()
        window_name = 'Age: ' + str(label) + ' Image ID: ' + str(image_idx)
        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, 40, 30)
        cv2.imshow(window_name, img)

        # First Number

        k = cv2.waitKey(0)

        if k == 48:
            number1 = 0
        elif k == 49:
            number1 = 1
        elif k == 50:
            number1 = 2
        elif k == 51:
            number1 = 3
        elif k == 52:
            number1 = 4
        elif k == 53:
            number1 = 5
        elif k == 54:
            number1 = 6
        elif k == 55:
            number1 = 7
        elif k == 56:
            number1 = 8
        elif k == 57:
            number1 = 9
        elif k == 27: # "escape" key
            break
        elif k == 83: # down arrow key:
            image_idx += 1
            continue
        elif k == 81: # left arrow:
            image_idx -= 1
            continue
        else:
            continue

        k = cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Second number

        if k == 48:
            number2 = 0
            annon_dict[img_id][annon]['age'] = int(str(number1) + str(number2))
            image_idx += 1
        elif k == 49:
            number2 = 1
            annon_dict[img_id][annon]['age'] = int(str(number1) + str(number2))
            image_idx += 1
        elif k == 50:
            number2 = 2
            annon_dict[img_id][annon]['age'] = int(str(number1) + str(number2))
            image_idx += 1
        elif k == 51:
            number2 = 3
            annon_dict[img_id][annon]['age'] = int(str(number1) + str(number2))
            image_idx += 1
        elif k == 52:
            number2 = 4
            annon_dict[img_id][annon]['age'] = int(str(number1) + str(number2))
            image_idx += 1
        elif k == 53:
            number2 = 5
            annon_dict[img_id][annon]['age'] = int(str(number1) + str(number2))
            image_idx += 1
        elif k == 54:
            number2 = 6
            annon_dict[img_id][annon]['age'] = int(str(number1) + str(number2))
            image_idx += 1
        elif k == 55:
            number2 = 7
            annon_dict[img_id][annon]['age'] = int(str(number1) + str(number2))
            image_idx += 1
        elif k == 56:
            number2 = 8
            annon_dict[img_id][annon]['age'] = int(str(number1) + str(number2))
            image_idx += 1
        elif k == 57:
            number2 = 9
            annon_dict[img_id][annon]['age'] = int(str(number1) + str(number2))
            image_idx += 1
        elif k == 27:  # "escape" key
            break
        elif k == 83:  # down arrow key:
            image_idx += 1
            continue
        elif k == 81:  # left arrow:
            image_idx -= 1
            continue
        else:
            continue


    r = json.dumps(annon_dict, indent=4)
    with open(os.path.join(output_dir, 'annotations', 'filtered_annotations_15000_labeled.json'), 'w') as f:
        f.write(r)


def filter_face_sizes(data_dir, data_type, output_dir):

    # get path for all faces
    images = [os.path.join(output_dir, 'images', image) for image in os.listdir(os.path.join(output_dir, 'images')) if
              'jpg' in image]

    # reads original json file if labeling has not started. Otherwise, open the json file that already modify some labels.
    with open(os.path.join(data_dir, 'annotations', '%s_400.json' % data_type), 'r') as f:
        annon_dict = json.loads(f.read())

    new_annon_dict = {}
    for idx, image in enumerate(images):
        img_id, _, _ = os.path.basename(image)[:-4].split('_')
        new_annon_dict[img_id] = {}

    for idx, image in enumerate(images):
        img_id, annon, _ = os.path.basename(image)[:-4].split('_')
        new_annon_dict[img_id][annon] = {}
        new_annon_dict[img_id][annon]['age'] = annon_dict[img_id][annon]['age_gender_pred']['predicted_ages']
        new_annon_dict[img_id][annon]['gender'] = annon_dict[img_id][annon]['age_gender_pred']['predicted_genders']

    # saves dictionary in a json file1
    r = json.dumps(new_annon_dict, indent=4)
    with open(os.path.join(output_dir, 'annotations', 'filtered_annotations_15000.json'), 'w') as f:
        f.write(r)

if __name__ == '__main__':
    data_dir = '/home/giancarlo/Documents/custom_dataset_final_results/data'
    data_type = 'train2017'
    output_dir = '/home/giancarlo/Documents/Age-test/custom_15000'
    age_labeling(output_dir, image_idx=155)
    #filter_face_sizes(data_dir, data_type, output_dir)