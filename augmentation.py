import os
import cv2
import random
from shutil import copyfile, rmtree
from tqdm import tqdm
import albumentations as A

def augment_and_save(image_path, label_path, output_image_dir, output_label_dir, angle):
    image = cv2.imread(image_path)
    img_size = image.shape[:2]

    with open(label_path, 'r') as f:
        lines = f.readlines()

    bboxes = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1]) * img_size[1]
        y_center = float(parts[2]) * img_size[0]
        bbox_width = float(parts[3]) * img_size[1]
        bbox_height = float(parts[4]) * img_size[0]
        x_min = x_center - bbox_width / 2
        y_min = y_center - bbox_height / 2
        x_max = x_center + bbox_width / 2
        y_max = y_center + bbox_height / 2
        bboxes.append([x_min, y_min, x_max, y_max, class_id])

    transform = A.Compose([
        A.Rotate(limit=(angle, angle), p=1.0, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    transformed = transform(image=image, bboxes=bboxes, class_labels=[bbox[-1] for bbox in bboxes])
    transformed_img = transformed['image']
    transformed_bboxes = transformed['bboxes']

    base_name = os.path.basename(image_path)
    rotated_image_path = os.path.join(output_image_dir, f"{os.path.splitext(base_name)[0]}_{angle}.jpg")
    rotated_label_path = os.path.join(output_label_dir, f"{os.path.splitext(base_name)[0]}_{angle}.txt")

    cv2.imwrite(rotated_image_path, transformed_img)
    with open(rotated_label_path, 'w') as f:
        for bbox in transformed_bboxes:
            x_min, y_min, x_max, y_max, class_id = bbox
            x_center = (x_min + x_max) / 2 / img_size[1]
            y_center = (y_min + y_max) / 2 / img_size[0]
            bbox_width = (x_max - x_min) / img_size[1]
            bbox_height = (y_max - y_min) / img_size[0]
            f.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

def process_and_augment_images(image_list, output_image_dir, output_label_dir, angles=[90, 180, 270]):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    for img_path, label_path in tqdm(image_list, desc="Processing images and labels"):
        if not os.path.exists(img_path) or not os.path.exists(label_path):
            print(f"Warning: Image or label file {img_path} or {label_path} not found. Skipping.")
            continue

        augment_and_save(img_path, label_path, output_image_dir, output_label_dir, 0)  # Save original
        for angle in angles:
            augment_and_save(img_path, label_path, output_image_dir, output_label_dir, angle)

def downsample_dataset(image_dir, label_dir, target_count):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    combined = list(zip(image_files, label_files))
    random.shuffle(combined)
    downsampled = combined[:target_count]

    temp_image_dir = image_dir + '_temp'
    temp_label_dir = label_dir + '_temp'
    os.makedirs(temp_image_dir, exist_ok=True)
    os.makedirs(temp_label_dir, exist_ok=True)

    for image_file, label_file in downsampled:
        copyfile(os.path.join(image_dir, image_file), os.path.join(temp_image_dir, image_file))
        copyfile(os.path.join(label_dir, label_file), os.path.join(temp_label_dir, label_file))

    rmtree(image_dir)
    rmtree(label_dir)
    os.rename(temp_image_dir, image_dir)
    os.rename(temp_label_dir, label_dir)

def infer_image(image, bboxes, output_path):
    for bbox in bboxes:
        class_id, x_center, y_center, width, height = bbox
        x_min = int((x_center - width / 2) * image.shape[1])
        y_min = int((y_center - height / 2) * image.shape[0])
        x_max = int((x_center + width / 2) * image.shape[1])
        y_max = int((y_center + height / 2) * image.shape[0])
        color = (0, 255, 0) if class_id == 0 else (0, 0, 255)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.imwrite(output_path, image)

def perform_inference(image_dir, label_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    for image_file in tqdm(image_files, desc="Performing inference"):
        img_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, image_file.replace('.jpg', '.txt'))
        if not os.path.exists(label_path):
            print(f"Warning: Label file {label_path} not found. Skipping this image.")
            continue
        image = cv2.imread(img_path)

        with open(label_path, 'r') as f:
            lines = f.readlines()

        bboxes = []
        for line in lines:
            parts = line.strip().split()
            bboxes.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])

        infer_image(image, bboxes, os.path.join(output_dir, image_file))

# Define paths
base_dir = 'dataset'
new_base_dir = 'aug_dataset'
train_image_dir = os.path.join(base_dir, 'images/train')
val_image_dir = os.path.join(base_dir, 'images/val')
train_label_dir = os.path.join(base_dir, 'labels/train')
val_label_dir = os.path.join(base_dir, 'labels/val')

# Set flag for inference
do_inference = True  # Set this flag to False if you do not want to perform inference

# Get list of images and labels
train_images = [(os.path.join(train_image_dir, img), os.path.join(train_label_dir, img.replace('.jpg', '.txt')))
                for img in os.listdir(train_image_dir) if img.endswith('.jpg')]
val_images = [(os.path.join(val_image_dir, img), os.path.join(val_label_dir, img.replace('.jpg', '.txt')))
              for img in os.listdir(val_image_dir) if img.endswith('.jpg')]

all_images = train_images + val_images

closed_door_ls = []
opened_door_ls = []

for img_path, label_path in tqdm(all_images, desc="Classifying images"):
    if not os.path.exists(img_path) or not os.path.exists(label_path):
        print(f"Warning: Image or label file {img_path} or {label_path} not found. Skipping.")
        continue

    with open(label_path, 'r') as f:
        first_line = f.readline().strip()
        if first_line and first_line.split()[0] == '0':
            closed_door_ls.append((img_path, label_path))
        elif first_line and first_line.split()[0] == '1':
            opened_door_ls.append((img_path, label_path))

print(len(closed_door_ls), len(opened_door_ls))

# Define output directories for augmented data
augmented_image_dir = os.path.join(new_base_dir, 'images')
augmented_label_dir = os.path.join(new_base_dir, 'labels')

# Augment the closed door images
process_and_augment_images(closed_door_ls, augmented_image_dir, augmented_label_dir)

# Downsample the augmented closed door images
target_count = 2 * 4 * len(opened_door_ls)
#downsample_dataset(augmented_image_dir, augmented_label_dir, target_count)
num_augmented_images = len([f for f in os.listdir(augmented_image_dir) if f.endswith('.jpg')])
print(f"Number of images in {augmented_image_dir}: {num_augmented_images}")

# Augment the opened door images
process_and_augment_images(opened_door_ls, augmented_image_dir, augmented_label_dir)

# Count the number of files in the augmented directories
num_augmented_images = len([f for f in os.listdir(augmented_image_dir) if f.endswith('.jpg')])
num_augmented_labels = len([f for f in os.listdir(augmented_label_dir) if f.endswith('.txt')])

print(f"Number of images in {augmented_image_dir}: {num_augmented_images}")
print(f"Number of labels in {augmented_label_dir}: {num_augmented_labels}")

# Perform inference if the flag is set
if do_inference:
    inference_result_dir = os.path.join(new_base_dir, 'inference_result')
    perform_inference(augmented_image_dir, augmented_label_dir, inference_result_dir)

# Continue with further processing if needed
