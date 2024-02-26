import json
import os
import shutil
import pathlib

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict

import tensorflow as tf

INPUT_SIZE = 416
CLASSES = 5
BATCH_SIZE = 32


def load_data(data_dir_path: str, archive_name: str):
    archive_path = os.path.join(data_dir_path, archive_name)
    shutil.unpack_archive(archive_path, data_dir_path)

    annot_cat = {}
    img_cat = {}
    for split in ("train", "valid", "test"):
        data_dir = pathlib.Path(os.path.join(
            data_dir_path, split)).with_suffix('')

        # save annotations
        ann_adress = os.path.join(data_dir, "_annotations.coco.json")
        with open(ann_adress) as f:
            annotation = json.load(f)

        annot_cat[f"{split}"] = annotation

        image_files = list(data_dir.glob('*.jpg'))
        img_cat[f"{split}"] = image_files
        print(f"{split} data:", len(image_files))  # training images

    return annot_cat, img_cat


def draw_rectangle(image, box, color=(0, 255, 0), thickness=2):
    """
    Draws a rectangle on the image.

    Parameters:
        image (numpy.ndarray): The input image.
        top_left (tuple): Coordinates of the top-left corner of the rectangle (x, y).
        bottom_right (tuple): Coordinates of the bottom-right corner of the rectangle (x, y).
        color (tuple): Color of the rectangle in BGR format. Default is green (0, 255, 0).
        thickness (int): Thickness of the rectangle's edges. Default is 2.
    """
    # Convert floating-point coordinates to integer
    x_min, y_min, width, height = box
    top_left = (x_min, y_min)
    bottom_right = (x_min + width, y_min + height)

    # Create an array of points representing the rectangle
    rect_pts = np.array([[top_left, (bottom_right[0], top_left[1]), bottom_right,
                          (top_left[0], bottom_right[1])]], dtype=np.int32)

    # Draw rectangle on the image
    cv.polylines(image, [rect_pts], isClosed=True,
                 color=color, thickness=thickness)


def create_dataset(annotations, images, dir, frac):
    X = []
    Y = []

    arr_imgs = annotations.get_imgIds()
    random_indices = np.random.choice(
        len(arr_imgs), int(frac*len(arr_imgs)), replace=False)

    for id in [arr_imgs[i] for i in random_indices]:
        img = cv.imread(os.path.join(
            dir, images[id]["file_name"]), cv.IMREAD_GRAYSCALE)

        ann = annotations.load_anns(id)[0]
        class_id = ann['category_id']
        box = np.array(ann['bbox'], dtype=float)

        img = img.astype(float) / 255.
        box = np.asarray(box, dtype=float) / INPUT_SIZE
        label = np.append(box, class_id)

        X.append(img)
        Y.append(label)

    X = np.array(X)
    X = np.expand_dims(X, axis=3)
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    Y = tf.convert_to_tensor(Y, dtype=tf.float32)

    return tf.data.Dataset.from_tensor_slices((X, Y))


def format_instance(image, label):
    return image, (tf.one_hot(int(label[4]), CLASSES), [label[0], label[1], label[2], label[3]])


def tune_training_ds(dataset):
    dataset = dataset.map(format_instance, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1024, reshuffle_each_iteration=True)
    dataset = dataset.repeat()  # The dataset be repeated indefinitely.
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def tune_validation_ds(dataset):
    dataset = dataset.map(format_instance, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat()
    return dataset


def plot_batch(dataset):
    plt.figure(figsize=(20, 10))
    colors_rgb = [  # corresponds to labels 1,..,5
        (255, 0, 0),      # Red
        (0, 0, 255),      # Blue
        (0, 255, 0),      # Green
        (255, 255, 0),    # Yellow
        (255, 165, 0)     # Orange
    ]

    for images, labels in dataset.take(1):
        for i in range(BATCH_SIZE):
            ax = plt.subplot(4, BATCH_SIZE//4, i + 1)
            label = labels[0][i]
            box = (labels[1][i] * INPUT_SIZE)
            box = tf.cast(box, tf.int32)

            image = images[i].numpy().astype("float") * 255.0
            image = image.astype(np.uint8)
            image_color = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

            index = tf.argmax(label).numpy()
            cv.rectangle(image_color, box.numpy(), colors_rgb[index], 2)

            plt.imshow(image_color)
            plt.axis("off")


class COCOParser:
    def __init__(self, anns_file, imgs_dir):
        with open(os.path.join(imgs_dir, anns_file), 'r') as f:
            coco = json.load(f)

        self.annIm_dict = defaultdict(list)
        self.cat_dict = {}
        self.annId_dict = {}
        self.im_dict = {}
        self.licenses_dict = {}

        for ann in coco['annotations']:
            self.annIm_dict[ann['image_id']].append(ann)
            self.annId_dict[ann['id']] = ann
        for img in coco['images']:
            self.im_dict[img['id']] = img
        for cat in coco['categories']:
            self.cat_dict[cat['id']] = cat
        for license in coco['licenses']:
            self.licenses_dict[license['id']] = license

    def get_imgIds(self):
        return list(self.im_dict.keys())

    def get_annIds(self, im_ids):
        im_ids = im_ids if isinstance(im_ids, list) else [im_ids]
        return [ann['id'] for im_id in im_ids for ann in self.annIm_dict[im_id]]

    def load_anns(self, ann_ids):
        ann_ids = ann_ids if isinstance(ann_ids, list) else [ann_ids]
        return [self.annId_dict[ann_id] for ann_id in ann_ids]

    def load_cats(self, class_ids):
        class_ids = class_ids if isinstance(class_ids, list) else [class_ids]
        return [self.cat_dict[class_id] for class_id in class_ids]

    def get_imgLicenses(self, im_ids):
        im_ids = im_ids if isinstance(im_ids, list) else [im_ids]
        lic_ids = [self.im_dict[im_id]["license"] for im_id in im_ids]
        return [self.licenses_dict[lic_id] for lic_id in lic_ids]


class DataStorage:
    def __init__(self, annot_cat, img_cat) -> None:
        self.img = img_cat
        self.annot = annot_cat

    def return_sample(self, sample_id: int, plot=True):
        sample_img_path = os.path.join(
            'data', 'train', self.annot["train"]["images"][sample_id]["file_name"])
        print("Path:", sample_img_path)

        if plot:
            sample_img = cv.imread(sample_img_path, cv.IMREAD_COLOR)
            plt.imshow(cv.cvtColor(sample_img, cv.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()

        sample_box = self.annot['train']["annotations"][sample_id]['bbox']
        print("Box:", sample_box)

        return sample_img_path, sample_box
