import os
import random
from shutil import copyfile, rmtree
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def read_and_classify_images_labels(image_dir, label_dir):
    closed_door = []
    opened_door = []

    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    for image_file in tqdm(image_files, desc="Classifying images and labels"):
        label_file = image_file.replace('.jpg', '.txt')
        label_path = os.path.join(label_dir, label_file)

        if not os.path.exists(label_path):
            print(f"Warning: Label file {label_path} not found. Skipping this image.")
            continue

        with open(label_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                class_id = int(first_line.split()[0])
                if class_id == 0:
                    closed_door.append((image_file, label_file))
                elif class_id == 1:
                    opened_door.append((image_file, label_file))

    return closed_door, opened_door

def downsample_data(closed_door, opened_door, factor=2):
    target_count = len(opened_door) * factor
    if len(closed_door) > target_count:
        closed_door = random.sample(closed_door, target_count)
    return closed_door

def split_data(data, split_ratios=(0.8, 0.1, 0.1)):
    train_data, val_test_data = train_test_split(data, test_size=(1 - split_ratios[0]), random_state=42)
    val_data, test_data = train_test_split(val_test_data, test_size=split_ratios[2] / (split_ratios[1] + split_ratios[2]), random_state=42)
    return train_data, val_data, test_data

def save_data(data, base_dir, subset_name):
    image_output_dir = os.path.join(base_dir, 'images', subset_name)
    label_output_dir = os.path.join(base_dir, 'labels', subset_name)
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)

    for image_file, label_file in tqdm(data, desc=f"Saving {subset_name} data"):
        copyfile(os.path.join(augmented_image_dir, image_file), os.path.join(image_output_dir, image_file))
        copyfile(os.path.join(augmented_label_dir, label_file), os.path.join(label_output_dir, label_file))

def check_image_label_pairs(image_dir, label_dir):
    missing_labels = []
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    for image_file in image_files:
        label_file = image_file.replace('.jpg', '.txt')
        label_path = os.path.join(label_dir, label_file)
        if not os.path.exists(label_path):
            missing_labels.append(image_file)
    return missing_labels

# Define paths
augmented_image_dir = 'aug_dataset/images'
augmented_label_dir = 'aug_dataset/labels'
base_output_dir = 'aug_dataset_new'

# Read and classify images and labels
closed_door, opened_door = read_and_classify_images_labels(augmented_image_dir, augmented_label_dir)
print(f"Total closed_door: {len(closed_door)}, Total opened_door: {len(opened_door)}")

# Downsample closed_door data
closed_door = downsample_data(closed_door, opened_door, factor=2)
print(f"Downsampled closed_door: {len(closed_door)}")

# Split data into train, val, and test sets
closed_train, closed_val, closed_test = split_data(closed_door, split_ratios=(0.8, 0.1, 0.1))
opened_train, opened_val, opened_test = split_data(opened_door, split_ratios=(0.8, 0.1, 0.1))

# Combine train, val, and test sets
train_data = closed_train + opened_train
val_data = closed_val + opened_val
test_data = closed_test + opened_test

# Save data to corresponding directories
save_data(train_data, base_output_dir, 'train')
save_data(val_data, base_output_dir, 'val')
save_data(test_data, base_output_dir, 'test')

# Check if every image has a corresponding label file
for subset in ['train', 'val', 'test']:
    subset_image_dir = os.path.join(base_output_dir, 'images', subset)
    subset_label_dir = os.path.join(base_output_dir, 'labels', subset)
    missing_labels = check_image_label_pairs(subset_image_dir, subset_label_dir)
    if missing_labels:
        print(f"Subset '{subset}' has missing label files for images: {missing_labels}")
    else:
        print(f"All images in subset '{subset}' have corresponding label files.")
