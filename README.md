# Initializing the dataset and classes

!pip install roboflow ultralytics matplotlib opencv-python albumentations==0.4.6 --quiet

#This command installs several important Python libraries:
#Roboflow: A library that connects to Roboflow, a platform for managing and augmenting datasets for computer vision tasks.
#Ultralytics: Provides a framework to use YOLOv8, a popular object detection and classification model.
#Matplotlib: A plotting library used to visualize data.
#OpenCV: An open-source computer vision library for image processing.
#Albumentations: A library for fast and flexible image augmentations, especially useful for tasks like object detection.


import os
import shutil
from collections import Counter, defaultdict

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.bbox_utils import normalize_bboxes, denormalize_bboxes

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from roboflow import Roboflow
rf = Roboflow(api_key="eGsEbW5nGpFjTU1Ml5Th")
project = rf.workspace("roboflow-gw7yv").project("website-screenshots")
version = project.version(1)
dataset = version.download("yolov8")

#Here, you’re importing various Python libraries and functions:
#os, shutil: For file handling and directory management.
#Counter, defaultdict: Handy tools from the collections module for counting and organizing data.
#Albumentations: The functions imported help in data augmentation, especially for bounding box handling in object detection.
#CV2 (OpenCV): Used for image processing tasks.
#Matplotlib and Pandas: For data visualization and handling.
#YAML: Useful for handling configuration files, especially if you want to save model or dataset settings.

# Example of use
image_dir = '/content/Website-Screenshots-1/train/images'  # Folders with images
label_dir = '/content/Website-Screenshots-1/train/labels'  # Folders with label files

with open('/content/Website-Screenshots-1/data.yaml', 'r') as f:
    data_yaml = yaml.safe_load(f)

class_names = data_yaml['names']


# Define the paths to the original folders and the new cloned folders
base_path = '/content/Website-Screenshots-1'  # Change this to your dataset's root directory
folders_to_clone = ['train', 'test', 'valid']

# Clone each folder
for folder in folders_to_clone:
    original_folder = os.path.join(base_path, folder)
    cloned_folder = os.path.join(base_path, f"{folder}_model")

    # Create the clone folder if it doesn't exist
    if not os.path.exists(cloned_folder):
        os.makedirs(cloned_folder)

    # Copy all files and subdirectories to the clone
    for item in os.listdir(original_folder):
        source = os.path.join(original_folder, item)
        destination = os.path.join(cloned_folder, item)
        if os.path.isdir(source):
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)

print("Cloning complete: created train_model, test_model, and valid_model folders with all files.")

# Stats

def get_class_name(class_idx):
    """
    Restituisce il nome della classe dato il suo indice, garantito dal dataset.
    """
    if 0 <= class_idx < len(class_names):
        return class_names[class_idx]
    else:
        return "Classe non valida"

# Esempio di lista di indici di classi
class_indices = [0, 1, 2, 3, 4, 5, 6, 7]  # Puoi cambiare questa lista con gli indici di classi che desideri

# Itera sugli indici di classe
for idx in class_indices:
    class_name = get_class_name(idx)  # Usa la funzione per ottenere il nome della classe
    print(f"Class {idx} corresponds to : {class_name}")

image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')])
label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

# Print a few samples
print(f"Number of images: {len(image_files)}")
print(f"Number of labels: {len(label_files)}")

def count_class_distribution(label_path, class_names):
    class_counts = Counter()

    # Iterate over label files and count each class occurrence
    label_files = sorted([f for f in os.listdir(label_path) if f.endswith('.txt')])

    for label_file in label_files:
        with open(os.path.join(label_path, label_file), 'r') as f:
            annotations = f.readlines()
            for annot in annotations:
                class_idx = int(annot.split()[0])  # Get the class index
                class_counts[class_idx] += 1  # Increment the count for this class

    # Convert class indices to class names
    sorted_class_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_class_names = [class_names[i] for i, _ in sorted_class_counts]
    sorted_counts = [count for _, count in sorted_class_counts]

    return sorted_class_names, sorted_counts

# Path to label files
label_path = '/content/Website-Screenshots-1/train/labels'

# Get the sorted class distribution
sorted_class_names, sorted_counts = count_class_distribution(label_path, class_names)

# Plot the sorted class distribution
plt.figure(figsize=(10, 6))
plt.bar(sorted_class_names, sorted_counts, color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.title('Sorted Class Distribution')
plt.xlabel('Class Label')
plt.ylabel('Frequency')
plt.show()
FIRST VISUALIZZATION OF IMAGES WITH BOXES
def plot_image_with_bboxes(image_path, label_path):
    # Leggi l'immagine
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Leggi le annotazioni
    h, w, _ = img.shape
    with open(label_path, 'r') as f:
        annotations = f.readlines()

    # Disegna le bounding boxes
    for annotation in annotations:
        class_idx, x_center, y_center, bbox_width, bbox_height = map(float, annotation.strip().split())
        # Converti il formato YOLO in coordinate pixel
        x_center *= w
        y_center *= h
        bbox_width *= w
        bbox_height *= h
        x_min = int(x_center - bbox_width / 2)
        y_min = int(y_center - bbox_height / 2)
        x_max = int(x_center + bbox_width / 2)
        y_max = int(y_center + bbox_height / 2)

        # Disegna la bounding box sull'immagine
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        class_name = get_class_name(int(class_idx))
        cv2.putText(img, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Mostra l'immagine con le bounding boxes
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Visualizza alcune immagini con le bounding boxes
for i in range(0, 10):
    image_sample = os.path.join(image_dir, image_files[i])
    label_sample = os.path.join(label_dir, label_files[i])
    plot_image_with_bboxes(image_sample, label_sample)


First visualizzation of bounding boxes descriptive statistics without standardization
import os
import matplotlib.pyplot as plt
import cv2

# List to store image dimensions
image_dimensions = []

# Iterate through image files and get dimensions
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    image_dimensions.append((width, height))

# Create a DataFrame for analysis
dimensions_df = pd.DataFrame(image_dimensions, columns=['Width', 'Height'])

# Summary statistics
print(dimensions_df.describe())
import os

bbox_area = []
bbox_classes = []
bbox_features = []



for label_file in label_files:
    with open(os.path.join(label_dir, label_file), 'r') as f:
        labels = f.readlines()
        for label in labels:
            parts = label.strip().split()
            if len(parts) == 5:
                class_idx = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                area = width * height
                bbox_area.append(area)  # Aggiunge l'area alla lista
                bbox_classes.append(class_idx)
                bbox_features.append([x_center, y_center, width, height, area])

DESCRIPTIVE STATISTICS OF BOUNDING BOXES:
from collections import defaultdict
import os
import pandas as pd

def bounding_box_statistics(copy_label_dir):
    """
    Calculates the statistics of bounding boxes for each class and returns a summary.
    """
    bbox_stats = defaultdict(list)  # Stores statistics for each class
    label_files = sorted([f for f in os.listdir(copy_label_dir) if f.endswith('.txt')])

    for label_file in label_files:
        label_path = os.path.join(copy_label_dir, label_file)

        with open(label_path, 'r') as f:
            labels = f.readlines()

        for label in labels:
            class_idx, x_center, y_center, width, height = map(float, label.strip().split())
            bbox_area = width * height
            bbox_stats[class_idx].append(bbox_area)

    # Calculate statistics
    stats_summary = {
        class_idx: {
            'Count': len(areas),
            'Mean': sum(areas) / len(areas) if areas else 0,
            'Min': min(areas) if areas else 0,
            'Max': max(areas) if areas else 0
        }
        for class_idx, areas in sorted(bbox_stats.items())  # Sort by class index
    }

    # Convert the summary to a DataFrame for tabular display
    stats_df = pd.DataFrame(stats_summary).T
    stats_df.index.name = 'Class'
    return stats_df

# Run the statistics calculation
stats = bounding_box_statistics(label_dir)

# Display the statistics as a sorted table
print(stats)

print(f"Bounding boxes number: {len(bbox_area)}")
DEEP STATISTICS:
import numpy as np

mean_area = np.mean(bbox_area)
median_area = np.median(bbox_area)
std_area = np.std(bbox_area)
min_area = np.min(bbox_area)
max_area = np.max(bbox_area)

print(f"Media: {mean_area}, Mediana: {median_area}, Deviazione Standard: {std_area}")
print(f"Minimo: {min_area}, Massimo: {max_area}")
# Visualizzazione della distribuzione delle aree delle bounding boxes
plt.figure(figsize=(10, 6))
plt.hist(bbox_area, bins=50, color='skyblue')
plt.title("Distribution of Bounding Box Areas")
plt.xlabel("Area")
plt.ylabel("Frequency")
plt.show()
import os

bbox_area = []
bbox_classes = []
bbox_features = []



for label_file in label_files:
    with open(os.path.join(label_dir, label_file), 'r') as f:
        labels = f.readlines()
        for label in labels:
            parts = label.strip().split()
            if len(parts) == 5:
                class_idx = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                area = width * height
                bbox_area.append(area)  # Aggiunge l'area alla lista
                bbox_classes.append(class_idx)
                bbox_features.append([x_center, y_center, width, height, area])

### *STATISTIC* TESTS:
SHAPIRO-WILK:
(Maximum size 5000 to have a consistent p-value)
import random
from scipy.stats import shapiro
# Assuming bbox_classes is populated as you process labels
class_counts = list(Counter(bbox_classes).values())

sample_size = 5000
bbox_area_sample = random.sample(bbox_area, sample_size)

stat, p_value = shapiro(bbox_area_sample)
print(f"P-value Shapiro-Wilk (campione di {sample_size}): {p_value}")

GINI COEFFICIENT:
Interpretation:

An index close to 0 indicates equity in the distribution of classes, close to 1 indicates strong disparity.
def gini_coefficient(class_counts):
    total = sum(class_counts)
    sorted_counts = sorted(class_counts)
    cumulative_counts = np.cumsum(sorted_counts)
    gini_index = 1 - 2 * sum(cumulative_counts) / (total * len(class_counts)) + 1 / len(class_counts)
    return gini_index

gini_index = gini_coefficient(class_counts)
print(f"Indice di Gini: {gini_index}")

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Prepara i dati per t-SNE
# Utilizziamo la lista bbox_features che abbiamo popolato in precedenza
# Ogni elemento in bbox_features è [x_center, y_center, width, height, area]

# Converti bbox_features in un array numpy
bbox_features_array = np.array(bbox_features)

# Utilizziamo solo le coordinate delle bounding boxes per t-SNE (escludiamo l'area)
bbox_coords = bbox_features_array[:, :4]

# Etichette di classe corrispondenti
bbox_labels = np.array(bbox_classes)

# Poiché t-SNE è computazionalmente costoso su grandi dataset, campioniamo un sottoinsieme
sample_size = 5000
if len(bbox_coords) > sample_size:
    indices = np.random.choice(len(bbox_coords), size=sample_size, replace=False)
    bbox_coords_sample = bbox_coords[indices]
    bbox_labels_sample = bbox_labels[indices]
else:
    bbox_coords_sample = bbox_coords
    bbox_labels_sample = bbox_labels

# Esegui t-SNE
tsne = TSNE(n_components=2, random_state=42)
bbox_coords_tsne = tsne.fit_transform(bbox_coords_sample)

Interpretation of the graph:
Explanation:
We extracted the coordinates of bounding boxes and class labels.
We sampled 5000 bounding boxes to make the computation manageable.
We ran t-SNE to reduce the data to 2 dimensions.
We visualized the results of t-SNE, coloring each point according to its class label.
This visualization can help you understand how the bounding boxes of different classes are distributed and whether there are overlaps or clusters.
From the graph we see that some classes are much more frequent than others:
Button (buttons) and Text (text) are the most common classes, with a frequency exceeding 20,000 annotated objects.
Label and Iframe, on the other hand, are much less frequent in the dataset, with less than 1,000 annotations.
This indicates that the dataset is unbalanced, meaning that some classes are overrepresented compared to others. This is an important detail for training the model, as it may have difficulty learning correctly to recognize the less represented classes.
Utility of the graph:

The class distribution graph is useful for diagnosing imbalance problems in the dataset.
If some classes are underrepresented, it may be necessary to adopt data balancing techniques (such as oversampling minority classes or undersampling majority classes) to improve model performance.

# Crea un grafico a dispersione dei risultati t-SNE
plt.figure(figsize=(12, 8))
scatter = plt.scatter(bbox_coords_tsne[:, 0], bbox_coords_tsne[:, 1], c=bbox_labels_sample, cmap='tab10', alpha=0.6)
plt.legend(handles=scatter.legend_elements()[0], labels=[get_class_name(i) for i in range(len(class_names))])
plt.title('Visualizzazione t-SNE delle Bounding Boxes')
plt.xlabel('Dimensione t-SNE 1')
plt.ylabel('Dimensione t-SNE 2')
plt.show()

# Data Cleaning
model_image_dir = '/content/Website-Screenshots-1/train_model/images'
model_label_dir = '/content/Website-Screenshots-1/train_model/labels'
# Load class labels from data.yaml
with open('/content/Website-Screenshots-1/data.yaml', 'r') as f:
    data_yaml = yaml.safe_load(f)
class_names = data_yaml['names']
 ## *Label Validation*
def validate_dataset_coordinates(labels_path):
    invalid_files = []
    for label_file in os.listdir(labels_path):
        with open(os.path.join(labels_path, label_file), 'r') as f:
            for line in f:
                class_idx, x_center, y_center, width, height = map(float, line.strip().split())
                if not all(0 < coord <= 1 for coord in [x_center, y_center, width, height]):
                    invalid_files.append(label_file)
                    break
    return invalid_files

invalid_labels = validate_dataset_coordinates(model_label_dir)
if invalid_labels:
    print(f"Found {len(invalid_labels)} files with invalid coordinates:")
    for file in invalid_labels[:10]:  # Print first 10 for brevity
        print(file)
    print("Consider cleaning or correcting these files before proceeding.")
else:
    print("All coordinates in the original dataset are valid.")
def clean_bounding_boxes_on_copy(copy_image_dir, copy_label_dir, min_threshold=0.001, max_threshold=0.9):
    """
    Filters bounding boxes that are too small or too large based on the thresholds.
    Deletes both the images and the corresponding label files if no valid bounding boxes remain,
    but works only on the copy directories.
    """
    cleaned_labels = 0
    total_labels = 0

    # Sort files to ensure matching
    image_files = sorted(os.listdir(copy_image_dir))
    label_files = sorted(os.listdir(copy_label_dir))

    for image_file, label_file in zip(image_files, label_files):
        image_path = os.path.join(copy_image_dir, image_file)
        label_path = os.path.join(copy_label_dir, label_file)

        # Check if the image and label file match
        image_base = os.path.splitext(image_file)[0]
        label_base = os.path.splitext(label_file)[0]

        if image_base != label_base:
            print(f"Warning: {image_file} and {label_file} do not match.")
            continue

        # Read the label file
        with open(label_path, 'r') as f:
            labels = f.readlines()

        new_labels = []
        for label in labels:
            total_labels += 1
            class_idx, x_center, y_center, width, height = map(float, label.strip().split())
            bbox_area = width * height

            # Filter based on the set thresholds
            if min_threshold <= bbox_area <= max_threshold:
                new_labels.append(label)
            else:
                cleaned_labels += 1

        # If there are no valid bounding boxes, delete both the image and the label file
        if len(new_labels) == 0:
            os.remove(image_path)
            os.remove(label_path)
            print(f"Deleted {image_file} and {label_file} due to lack of valid bounding boxes.")
        else:
            # Rewrite the label file only with valid bounding boxes
            with open(label_path, 'w') as f:
                for new_label in new_labels:
                    f.write(new_label)

    print(f"Total bounding boxes processed: {total_labels}")
    print(f"Bounding boxes removed: {cleaned_labels}")

# Run data cleaning on the copy directories
clean_bounding_boxes_on_copy(model_image_dir, model_label_dir, min_threshold=0.001, max_threshold=0.9)
image_cnt = sorted([f for f in os.listdir(model_image_dir) if f.endswith('.jpg') or f.endswith('.png')])
label_cnt = sorted([f for f in os.listdir(model_label_dir) if f.endswith('.txt')])

print(f"Number of images: {len(image_cnt)}")
print(f"Number of labels: {len(label_cnt)}")
def count_class_distribution(label_path, class_names):
    class_counts = Counter()

    # Iterate over label files and count each class occurrence
    label_files = sorted([f for f in os.listdir(label_path) if f.endswith('.txt')])

    for label_file in label_files:
        with open(os.path.join(label_path, label_file), 'r') as f:
            annotations = f.readlines()
            for annot in annotations:
                class_idx = int(annot.split()[0])  # Get the class index
                class_counts[class_idx] += 1  # Increment the count for this class

    # Convert class indices to class names
    sorted_class_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_class_names = [class_names[i] for i, _ in sorted_class_counts]
    sorted_counts = [count for _, count in sorted_class_counts]

    return sorted_class_names, sorted_counts

# Path to label files
label_path = '/content/Website-Screenshots-1/train_model/labels'

# Get the sorted class distribution
sorted_class_names, sorted_counts = count_class_distribution(label_path, class_names)

# Plot the sorted class distribution
plt.figure(figsize=(10, 6))
plt.bar(sorted_class_names, sorted_counts, color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.title('Sorted Class Distribution After Bounding Box Cleaning')
plt.xlabel('Class Label')
plt.ylabel('Frequency')
plt.show()
## Under Sampling
import random
# Define classes to undersample (e.g., button, text, image)
high_freq_classes = ['button', 'text', 'image']  # Adjust based on your earlier analysis

# Define the undersampling ratio for each class
undersample_ratios = {
    'button': 0.5,  # Keep 50% of button instances
    'text': 0.6,    # Keep 60% of text instances
    'image': 0.7    # Keep 70% of image instances
}


# Function to undersample the high-represented classes
def undersample_labels(label_path, class_names, high_freq_classes, undersample_ratios):
    label_files = sorted([f for f in os.listdir(label_path) if f.endswith('.txt')])

    for label_file in label_files:
        label_full_path = os.path.join(label_path, label_file)

        with open(label_full_path, 'r') as f:
            annotations = f.readlines()

        updated_annotations = []

        for annot in annotations:
            class_idx, x_center, y_center, bbox_width, bbox_height = map(float, annot.split())
            class_name = class_names[int(class_idx)]

            # If it's a high-frequency class, apply undersampling
            if class_name in high_freq_classes:
                keep_probability = undersample_ratios[class_name]

                # Decide whether to keep this bounding box based on the probability
                if random.random() < keep_probability:
                    updated_annotations.append(annot)
            else:
                # If it's not a high-frequency class, keep it as is
                updated_annotations.append(annot)

        # Rewrite the label file with the updated annotations
        with open(label_full_path, 'w') as f:
            for annot in updated_annotations:
                f.write(annot)

        # Print some progress information
        print(f"Processed {label_file}: {len(updated_annotations)} remaining annotations after undersampling.")

# Call the undersample function on your dataset
undersample_labels('/content/Website-Screenshots-1/train_model/labels', class_names, high_freq_classes, undersample_ratios)

def count_class_distribution(label_path, class_names):
    class_counts = Counter()

    # Iterate over label files and count each class occurrence
    label_files = sorted([f for f in os.listdir(label_path) if f.endswith('.txt')])

    for label_file in label_files:
        with open(os.path.join(label_path, label_file), 'r') as f:
            annotations = f.readlines()
            for annot in annotations:
                class_idx = int(annot.split()[0])  # Get the class index
                class_counts[class_idx] += 1  # Increment the count for this class

    # Convert class indices to class names
    sorted_class_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_class_names = [class_names[i] for i, _ in sorted_class_counts]
    sorted_counts = [count for _, count in sorted_class_counts]

    return sorted_class_names, sorted_counts

# Path to label files
label_path = '/content/Website-Screenshots-1/train_model/labels'

# Get the sorted class distribution
sorted_class_names, sorted_counts = count_class_distribution(label_path, class_names)

# Plot the sorted class distribution
plt.figure(figsize=(10, 6))
plt.bar(sorted_class_names, sorted_counts, color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.title('Sorted Class Distribution After Undersampling')
plt.xlabel('Class Label')
plt.ylabel('Frequency')
plt.show()
##Data Augmentation
import yaml

# Path to the data.yaml file
yaml_path = '/content/Website-Screenshots-1/data.yaml'  # Update this path if necessary

# Load the YAML file
with open(yaml_path, 'r') as file:
    data_yaml = yaml.safe_load(file)

# Extract class names and enumerate their indexes
class_names = data_yaml.get('names', [])
class_index_mapping = {index: name for index, name in enumerate(class_names)}

# Display the class names with indexes
print("Class Index Mapping:")
for index, name in class_index_mapping.items():
    print(f"Index: {index}, Class: {name}")

import os
import cv2
import random
import numpy as np
from PIL import Image, ImageEnhance

# Helper function to apply augmentations to an image
def augment_image(image):
    augmentations = [ImageEnhance.Color, ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Sharpness]
    # Apply random enhancement
    enhancer = random.choice(augmentations)
    factor = random.uniform(0.7, 1.3)  # Factor for augmentation
    image = enhancer(image).enhance(factor)

    # Optionally, add more augmentations like flipping, scaling, rotating, etc.
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image

# Function to crop images around the bounding box of underrepresented classes
def crop_and_augment(images_path, labels_path, target_classes, min_size=(50, 50), augmentation_factor=5):
    augmented_count = 0
    for label_file in os.listdir(labels_path):
        label_path = os.path.join(labels_path, label_file)
        image_path = os.path.join(images_path, label_file.replace('.txt', '.jpg'))

        if not os.path.exists(image_path):
            continue

        # Read the image
        image = Image.open(image_path)
        image_w, image_h = image.size

        # Read the label file
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            class_idx, x_center, y_center, width, height = map(float, line.strip().split())
            class_idx = int(class_idx)

            if class_idx in target_classes:
                # Convert normalized coordinates back to absolute coordinates
                x_center_abs = x_center * image_w
                y_center_abs = y_center * image_h
                width_abs = width * image_w
                height_abs = height * image_h

                # Check if the bounding box is greater than the minimum size
                if width_abs > min_size[0] and height_abs > min_size[1]:
                    # Calculate the bounding box (xmin, ymin, xmax, ymax)
                    xmin = int(x_center_abs - width_abs / 2)
                    ymin = int(y_center_abs - height_abs / 2)
                    xmax = int(x_center_abs + width_abs / 2)
                    ymax = int(y_center_abs + height_abs / 2)

                    # Crop the image around the bounding box
                    cropped_image = image.crop((xmin, ymin, xmax, ymax))

                    # Apply augmentation to the cropped image multiple times
                    for i in range(augmentation_factor):
                        augmented_image = augment_image(cropped_image)

                        # Save the augmented image
                        new_image_name = label_file.replace('.txt', f'_aug_{i}.jpg')
                        new_image_path = os.path.join(images_path, new_image_name)
                        augmented_image.save(new_image_path)

                        # Adjust the label for the new augmented image
                        new_label_name = label_file.replace('.txt', f'_aug_{i}.txt')
                        new_label_path = os.path.join(labels_path, new_label_name)

                        # Since the image is cropped, the bounding box becomes the whole image (center = 0.5, 0.5 and width = 1, height = 1)
                        new_x_center, new_y_center, new_width, new_height = 0.5, 0.5, 1.0, 1.0

                        with open(new_label_path, 'w') as new_label_file:
                            new_label_file.write(f"{class_idx} {new_x_center} {new_y_center} {new_width} {new_height}\n")

                        augmented_count += 1

    print(f"Total augmented images: {augmented_count}")

# Define paths
dataset_path = '/content/Website-Screenshots-1'
train_images_path = os.path.join(dataset_path, 'train_model', 'images')
train_labels_path = os.path.join(dataset_path, 'train_model', 'labels')

# Define the target underrepresented classes (for example, classes with < 1000 occurrences)
underrepresented_classes = [1, 3, 5]  # Replace with actual class indices

# Perform cropping and augmentation
crop_and_augment(train_images_path, train_labels_path, underrepresented_classes, min_size=(100, 100), augmentation_factor=5)

def count_class_distribution(label_path, class_names):
    class_counts = Counter()

    # Iterate over label files and count each class occurrence
    label_files = sorted([f for f in os.listdir(label_path) if f.endswith('.txt')])

    for label_file in label_files:
        with open(os.path.join(label_path, label_file), 'r') as f:
            annotations = f.readlines()
            for annot in annotations:
                class_idx = int(annot.split()[0])  # Get the class index
                class_counts[class_idx] += 1  # Increment the count for this class

    # Convert class indices to class names
    sorted_class_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_class_names = [class_names[i] for i, _ in sorted_class_counts]
    sorted_counts = [count for _, count in sorted_class_counts]

    return sorted_class_names, sorted_counts

# Path to label files
label_path = '/content/Website-Screenshots-1/train_model/labels'

# Get the sorted class distribution
sorted_class_names, sorted_counts = count_class_distribution(label_path, class_names)

# Plot the sorted class distribution
plt.figure(figsize=(10, 6))
plt.bar(sorted_class_names, sorted_counts, color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.title('Sorted Class Distribution After Augmentation')
plt.xlabel('Class Label')
plt.ylabel('Frequency')
plt.show()
image_cnt = sorted([f for f in os.listdir(model_image_dir) if f.endswith('.jpg') or f.endswith('.png')])
label_cnt = sorted([f for f in os.listdir(model_label_dir) if f.endswith('.txt')])

print(f"Number of images: {len(image_cnt)}")
print(f"Number of labels: {len(label_cnt)}")
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def find_and_plot_augmented_images(images_path, labels_path, num_images=10):
    augmented_images = []

    # Traverse the directory and collect augmented image paths and corresponding label data
    for label_file in os.listdir(labels_path):
        if '_aug_' in label_file:
            image_file = label_file.replace('.txt', '.jpg')
            image_path = os.path.join(images_path, image_file)
            label_path = os.path.join(labels_path, label_file)

            if os.path.exists(image_path):
                # Open the image and label file
                image = Image.open(image_path)

                with open(label_path, 'r') as f:
                    line = f.readline().strip()
                    if line:
                        class_idx, x_center, y_center, width, height = map(float, line.split())
                        augmented_images.append((image, class_idx, x_center, y_center, width, height))

            # Stop after collecting the desired number of images
            if len(augmented_images) >= num_images:
                break

    # Plot the augmented images with bounding boxes
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for idx, (image, class_idx, x_center, y_center, width, height) in enumerate(augmented_images[:num_images]):
        row, col = divmod(idx, 5)
        ax = axes[row, col]
        ax.imshow(image)

        # Calculate absolute bounding box coordinates
        image_w, image_h = image.size
        bbox_xmin = int((x_center - width / 2) * image_w)
        bbox_ymin = int((y_center - height / 2) * image_h)
        bbox_xmax = int((x_center + width / 2) * image_w)
        bbox_ymax = int((y_center + height / 2) * image_h)

        # Draw the bounding box
        bbox = patches.Rectangle((bbox_xmin, bbox_ymin), bbox_xmax - bbox_xmin, bbox_ymax - bbox_ymin,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(bbox)
        ax.set_title(f"Class: {int(class_idx)}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Define paths
dataset_path = '/content/Website-Screenshots-1'
train_images_path = os.path.join(dataset_path, 'train_model', 'images')
train_labels_path = os.path.join(dataset_path, 'train_model', 'labels')

# Find and plot augmented images
find_and_plot_augmented_images(train_images_path, train_labels_path, num_images=10)

import os

# Get the base filenames without extensions for both images and labels
model_image_files = {os.path.splitext(f)[0] for f in os.listdir(model_image_dir)}
model_label_files = {os.path.splitext(f)[0] for f in os.listdir(model_label_dir)}

# Find extra label files
extra_labels = model_label_files - model_image_files

# Delete each extra label file
for label in extra_labels:
    label_path = os.path.join(label_dir, label + '.txt')
    os.remove(label_path)
    print(f"Deleted extra label file: {label_path}")

print(f"Total extra labels deleted: {len(extra_labels)}")

def count_class_distribution(label_path, class_names):
    class_counts = Counter()

    # Iterate over label files and count each class occurrence
    label_files = sorted([f for f in os.listdir(label_path) if f.endswith('.txt')])

    for label_file in label_files:
        with open(os.path.join(label_path, label_file), 'r') as f:
            annotations = f.readlines()
            for annot in annotations:
                class_idx = int(annot.split()[0])  # Get the class index
                class_counts[class_idx] += 1  # Increment the count for this class

    # Convert class indices to class names
    sorted_class_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_class_names = [class_names[i] for i, _ in sorted_class_counts]
    sorted_counts = [count for _, count in sorted_class_counts]

    return sorted_class_names, sorted_counts

# Path to label files
label_path = '/content/Website-Screenshots-1/train_model/labels'

# Get the sorted class distribution
sorted_class_names, sorted_counts = count_class_distribution(label_path, class_names)

# Plot the sorted class distribution
plt.figure(figsize=(10, 6))
plt.bar(sorted_class_names, sorted_counts, color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.title('Sorted Class Distribution After Bounding Box Cleaning')
plt.xlabel('Class Label')
plt.ylabel('Frequency')
plt.show()
## BOUNDING BOXES DESCRIPTIVE STATISTICS AFTER DATA CLEANING
from collections import defaultdict
import os
import pandas as pd

def bounding_box_statistics(copy_label_dir):
    """
    Calculates the statistics of bounding boxes for each class and returns a summary.
    """
    bbox_stats = defaultdict(list)  # Stores statistics for each class
    label_files = sorted([f for f in os.listdir(copy_label_dir) if f.endswith('.txt')])

    for label_file in label_files:
        label_path = os.path.join(copy_label_dir, label_file)

        with open(label_path, 'r') as f:
            labels = f.readlines()

        for label in labels:
            class_idx, x_center, y_center, width, height = map(float, label.strip().split())
            bbox_area = width * height
            bbox_stats[class_idx].append(bbox_area)

    # Calculate statistics
    stats_summary = {
        class_idx: {
            'Count': len(areas),
            'Mean': sum(areas) / len(areas) if areas else 0,
            'Min': min(areas) if areas else 0,
            'Max': max(areas) if areas else 0
        }
        for class_idx, areas in sorted(bbox_stats.items())  # Sort by class index
    }

    # Convert the summary to a DataFrame for tabular display
    stats_df = pd.DataFrame(stats_summary).T
    stats_df.index.name = 'Class'
    return stats_df

# Run the statistics calculation
stats = bounding_box_statistics(model_label_dir)

# Display the statistics as a sorted table
print(stats)

MINORITY CLASS DEFINITION:
# Classi minoritarie definite
minority_classes = [1, 3, 5]  # label = 6, iframe = 7, field = 5

import os
from collections import defaultdict

def find_images_with_class_distribution(copy_label_dir, low_classes, high_classes, max_class_0_7=4, min_class_1_3_5=2):
    """
    Trova immagini con un basso contenuto di classi 0 e 7 e un alto contenuto di classi 1, 3 e 5.
    Restituisce le immagini e le statistiche per ciascuna immagine.
    """
    selected_images = []
    class_distribution_stats = []

    label_files2 = sorted([f for f in os.listdir(copy_label_dir) if f.endswith('.txt')])

    for label_file in label_files2:
        label_path = os.path.join(copy_label_dir, label_file)

        # Conteggio delle classi per ogni immagine
        class_counts = defaultdict(int)

        with open(label_path, 'r') as f:
            labels = f.readlines()

        for label in labels:
            class_idx = int(label.strip().split()[0])
            class_counts[class_idx] += 1

        # Verifica delle condizioni
        if class_counts[0] <= max_class_0_7 and class_counts[7] <= max_class_0_7 :  # Condizione per classi dominanti
            if class_counts[1] >= min_class_1_3_5 or class_counts[3] >= min_class_1_3_5 or class_counts[5] >= min_class_1_3_5:  # Condizione per classi minoritarie
                selected_images.append(label_file)
                class_distribution_stats.append(class_counts)

    # Statistiche descrittive finali
    total_images = len(selected_images)
    total_class_counts = defaultdict(int)

    for class_stats in class_distribution_stats:
        for class_idx, count in class_stats.items():
            total_class_counts[class_idx] += count

    return selected_images, total_images, total_class_counts

# Impostazioni delle classi
low_classes = [0, 7, 2, 4, 6]  # Classi con basso contenuto (0 e 7)
high_classes = [1, 3, 5]  # Classi con alto contenuto (1, 3, 5)

# FILTER:
selected_images, total_images, total_class_counts = find_images_with_class_distribution(
    model_label_dir, low_classes, high_classes, max_class_0_7=4, min_class_1_3_5=2)

# Visualizza i risultati
print(f"Images selected: {total_images}")
print(f"Class distribution in the selected images:")
for class_idx, count in total_class_counts.items():
    print(f"  Classe {class_idx}: {count}")

AVERAGE number of labels on each image found
def calculate_class_averages(total_class_counts, total_images):
    """
    Calcola la media delle istanze di ciascuna classe per immagine.
    """
    class_averages = {class_idx: count / total_images for class_idx, count in total_class_counts.items()}
    return class_averages

# Calcola le medie
class_averages = calculate_class_averages(total_class_counts, total_images)

# Visualizza le medie
print(f"Mean number of data points per image in the class:")
for class_idx, avg in class_averages.items():
    print(f"  Classe {class_idx}: {avg:.2f} (mean per image)")

# Modeling
import os

# Define the paths for renaming
dataset_path = '/content/Website-Screenshots-1'
train_old_path = os.path.join(dataset_path, 'train_old')
train_model_path = os.path.join(dataset_path, 'train_model')
train_new_path = os.path.join(dataset_path, 'train')

# First, rename 'train' to 'train_old' if it exists
if os.path.exists(train_new_path):
    os.rename(train_new_path, train_old_path)
    print(f"Renamed 'train' to 'train_old'")

# Then, rename 'train_model' to 'train'
if os.path.exists(train_model_path):
    os.rename(train_model_path, train_new_path)
    print(f"Renamed 'train_model' to 'train'")

!pip install ultralytics
def validate_labels(labels_path, num_classes):
    invalid_files = []
    for label_file in os.listdir(labels_path):
        with open(os.path.join(labels_path, label_file), 'r') as f:
            for line in f:
                class_idx = int(line.strip().split()[0])
                if class_idx < 0 or class_idx >= num_classes:
                    invalid_files.append(label_file)
                    break
    return invalid_files

# Assuming you have the number of classes from your data.yaml
num_classes = len(class_names)
invalid_labels = validate_labels('/content/Website-Screenshots-1/train/labels', num_classes)

if invalid_labels:
    print(f"Found {len(invalid_labels)} files with invalid class indices.")
else:
    print("All labels are valid.")

from ultralytics import YOLO

# Load YOLOv8 model (you can use a pretrained model like 'yolov8n.pt' or 'yolov8s.pt')
model = YOLO('yolov8n.pt')
# Train the YOLOv8 model
results = model.train(data='/content/Website-Screenshots-1/data.yaml',
                      epochs=20,
                      imgsz=608,
                      batch=16,
                      lr0=0.0008,            # Initial learning rate
                      lrf=0.001,             # Final learning rate
                      weight_decay=0.0005,  # Weight decay
                      momentum=0.9,       # Momentum
                      augment=True,          # Enable augmentations
                      )
model_path = '/content/Website-Screenshots-1/5_epoch_model.pt'  # Define the path to save your model
model.save(model_path)
print(f"Model saved at {model_path}")
test_images_path = '/content/Website-Screenshots-1/test/images'
test_image_files = sorted([f for f in os.listdir(test_images_path) if f.endswith('.jpg') or f.endswith('.png')])

# Limit to the first 10 images
test_image_files = test_image_files[:10]
for image_file in test_image_files:
    image_path = os.path.join(test_images_path, image_file)

    # Make predictions
    reslist = model.predict(image_path)

    # Visualize predictions
    for res in reslist[:10]:
        img = res.orig_img  # Original image with predictions
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy.cpu().numpy().astype(int)[0]
            cls = int(box.cls[0].cpu().numpy())  # Class index
            conf = box.conf[0].cpu().numpy()  # Confidence score

            # Draw bounding boxes
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{model.names[cls]}: {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Show the image with predictions
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(image_file)
        plt.show()

# Evaluate the model on the validation set
val_res = model.val(data='/content/Website-Screenshots-1/data.yaml')
import seaborn as sns
import matplotlib.pyplot as plt

# Access the confusion matrix from the validation results
conf_matrix = val_res.confusion_matrix.matrix

# Class names (replace these with actual class names if needed)
class_names = ['button', 'field', 'heading', 'iframe', 'image', 'label', 'link', 'text']

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt=".1f", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

import numpy as np
import pandas as pd  # Import pandas for table creation

# Access the confusion matrix
conf_matrix = val_res.confusion_matrix.matrix

# Initialize lists to store precision, recall, and F1-score
precisions = []
recalls = []
f1_scores = []

# Class names (replace with actual class names if needed)
class_names = ['button', 'field', 'heading', 'iframe', 'image', 'label', 'link', 'text']

# Calculate metrics for each class
for i in range(len(class_names)):
    tp = conf_matrix[i, i]  # True positives
    fp = conf_matrix[:, i].sum() - tp  # False positives
    fn = conf_matrix[i, :].sum() - tp  # False negatives

    # Precision and recall calculations
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# Create a DataFrame to hold the results
results_df = pd.DataFrame({
    'Class': class_names,
    'Precision': precisions,
    'Recall': recalls,
    'F1-Score': f1_scores
})

# Set the Class column as the index
results_df.set_index('Class', inplace=True)

# Print the results in table format
print(results_df)

### Modeling With Data Merged and iframe removed
# Define the paths to the original folders and the new cloned folders
base_path = '/content/Website-Screenshots-1'  # Change this to your dataset's root directory
folders_to_clone = ['train', 'test', 'valid']

# Clone each folder
for folder in folders_to_clone:
    original_folder = os.path.join(base_path, folder)
    cloned_folder = os.path.join(base_path, f"{folder}_modelv2")

    # Create the clone folder if it doesn't exist
    if not os.path.exists(cloned_folder):
        os.makedirs(cloned_folder)

    # Copy all files and subdirectories to the clone
    for item in os.listdir(original_folder):
        source = os.path.join(original_folder, item)
        destination = os.path.join(cloned_folder, item)
        if os.path.isdir(source):
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)

print("Cloning complete: created train_model, test_model, and valid_model folders with all files.")
def remove_class_from_labels(copy_label_dir, class_to_remove):
    """
    Rimuove una classe specifica dai file di etichette.
    """
    label_files2 = sorted([f for f in os.listdir(copy_label_dir) if f.endswith('.txt')])

    for label_file in label_files2:
        label_path = os.path.join(copy_label_dir, label_file)

        # Leggi e filtra le etichette
        updated_labels = []
        with open(label_path, 'r') as f:
            labels = f.readlines()
            for label in labels:
                class_idx, x_center, y_center, width, height = map(float, label.strip().split())
                if int(class_idx) != class_to_remove:  # Rimuovi solo la classe specificata
                    updated_labels.append(f"{int(class_idx)} {x_center} {y_center} {width} {height}\n")

        # Sovrascrivi il file di etichette senza la classe rimossa
        with open(label_path, 'w') as f:
            f.writelines(updated_labels)

        print(f"Class {class_to_remove} removed from {label_file}")

# Rimuovere la classe 'iframe' (Classe 3)
remove_class_from_labels('/content/Website-Screenshots-1/train_modelv2/labels', class_to_remove=3)
remove_class_from_labels('/content/Website-Screenshots-1/test_modelv2/labels', class_to_remove=3)
remove_class_from_labels('/content/Website-Screenshots-1/valid_modelv2/labels', class_to_remove=3)


def merge_classes_in_labels(copy_label_dir, from_class_idx, to_class_idx):
    """
    Unisce due classi nel file delle etichette, sostituendo tutte le istanze della classe from_class_idx con to_class_idx.
    """
    label_files2 = sorted([f for f in os.listdir(copy_label_dir) if f.endswith('.txt')])

    for label_file in label_files2:
        label_path = os.path.join(copy_label_dir, label_file)

        # Leggi e modifica le etichette
        updated_labels = []
        with open(label_path, 'r') as f:
            labels = f.readlines()
            for label in labels:
                class_idx, x_center, y_center, width, height = map(float, label.strip().split())
                if int(class_idx) == from_class_idx:
                    class_idx = to_class_idx  # Sostituisci la classe
                updated_labels.append(f"{int(class_idx)} {x_center} {y_center} {width} {height}\n")

        # Sovrascrivi il file di etichette con le classi unite
        with open(label_path, 'w') as f:
            f.writelines(updated_labels)

        print(f"Classi {from_class_idx} unite con {to_class_idx} in {label_file}")

# Unire la classe 'field' (Classe 1) con 'label' (Classe 5)
merge_classes_in_labels('/content/Website-Screenshots-1/train_modelv2/labels', from_class_idx=5, to_class_idx=1)
merge_classes_in_labels('/content/Website-Screenshots-1/test_modelv2/labels', from_class_idx=5, to_class_idx=1)
merge_classes_in_labels('/content/Website-Screenshots-1/valid_modelv2/labels', from_class_idx=5, to_class_idx=1)

'''
import yaml

# Define the path to the data.yaml file
yaml_file_path = '/content/Website-Screenshots-1/data.yaml'

# Load the existing data.yaml file
with open(yaml_file_path, 'r') as file:
    data = yaml.safe_load(file)

# Modify the class names
# Remove 'iframe' (index 3) and merge 'field' (index 1) into 'label' (index 5)
data['names'] = [
    'button',    # index 0
    'label',     # 'field' merged into 'label' (index 1)
    'heading',   # index 2
    'image',     # index 3
    'link',      # index 4
    'text'       # index 5
]
data['nc'] = len(data['names'])  # Update the number of classes

# Save the updated data.yaml file
with open(yaml_file_path, 'w') as file:
    yaml.dump(data, file, default_flow_style=False)

print("data.yaml has been updated successfully.")
'''
import os

def get_unique_classes_from_labels(label_dir):
    """
    Get unique class indices from label files in the specified directory.

    :param label_dir: Directory containing the label files.
    :return: A set of unique class indices.
    """
    unique_classes = set()
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)

        with open(label_path, 'r') as f:
            labels = f.readlines()
            for label in labels:
                class_idx = int(label.strip().split()[0])  # Get the class index
                unique_classes.add(class_idx)  # Add to the set of unique classes

    return unique_classes

# Specify the directory containing the training labels
label_directory = '/content/Website-Screenshots-1/train_modelv2/labels'
unique_classes = get_unique_classes_from_labels(label_directory)

# Print the unique classes
print("Unique classes in the training dataset:", unique_classes)

import os

# Define the paths for renaming
dataset_path = '/content/Website-Screenshots-1'
train_old_path = os.path.join(dataset_path, 'train_old2')
train_model_path = os.path.join(dataset_path, 'train_modelv2')
train_new_path = os.path.join(dataset_path, 'train')

# First, rename 'train' to 'train_old' if it exists
if os.path.exists(train_new_path):
    os.rename(train_new_path, train_old_path)
    print(f"Renamed 'train' to 'train_old'")

# Then, rename 'train_model' to 'train'
if os.path.exists(train_model_path):
    os.rename(train_model_path, train_new_path)
    print(f"Renamed 'train_model' to 'train'")

from ultralytics import YOLO

# Load YOLOv8 model (you can use a pretrained model like 'yolov8n.pt' or 'yolov8s.pt')
model2 = YOLO('yolov8n.pt')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Train the YOLOv8 model
results2 = model2.train(data='/content/Website-Screenshots-1/data.yaml',
                      epochs=20,
                      imgsz=608,
                      batch=16,
                      lr0=0.0008,            # Initial learning rate
                      lrf=0.001,             # Final learning rate
                      weight_decay=0.0005,  # Weight decay
                      momentum=0.9,       # Momentum
                      augment=True,          # Enable augmentations
                      )
model_path = '/content/Website-Screenshots-1/5_epoch_model_merged.pt'  # Define the path to save your model
model2.save(model_path)
print(f"Model saved at {model_path}")
test_images_path = '/content/Website-Screenshots-1/test/images'
test_image_files = sorted([f for f in os.listdir(test_images_path) if f.endswith('.jpg') or f.endswith('.png')])

# Limit to the first 10 images
test_image_files = test_image_files[:10]
for image_file in test_image_files:
    image_path = os.path.join(test_images_path, image_file)

    # Make predictions
    reslist2 = model2.predict(image_path)

    # Visualize predictions
    for res in reslist2[:10]:
        img = res.orig_img  # Original image with predictions
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy.cpu().numpy().astype(int)[0]
            cls = int(box.cls[0].cpu().numpy())  # Class index
            conf = box.conf[0].cpu().numpy()  # Confidence score

            # Draw bounding boxes
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{model.names[cls]}: {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Show the image with predictions
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(image_file)
        plt.show()

# Evaluate the model on the validation set
val_res2 = model2.val(data='/content/Website-Screenshots-1/data.yaml')
import seaborn as sns
import matplotlib.pyplot as plt

# Access the confusion matrix from the validation results
conf_matrix = val_res2.confusion_matrix.matrix

# Class names (replace these with actual class names if needed)
class_names = ['button', 'field', 'heading', 'iframe', 'image', 'label', 'link', 'text']

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt=".1f", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

import numpy as np
import pandas as pd  # Import pandas for table creation

# Access the confusion matrix
conf_matrix = val_res2.confusion_matrix.matrix

# Initialize lists to store precision, recall, and F1-score
precisions = []
recalls = []
f1_scores = []

# Class names (replace with actual class names if needed)
class_names = ['button', 'field', 'heading', 'iframe', 'image', 'label', 'link', 'text']

# Calculate metrics for each class
for i in range(len(class_names)):
    tp = conf_matrix[i, i]  # True positives
    fp = conf_matrix[:, i].sum() - tp  # False positives
    fn = conf_matrix[i, :].sum() - tp  # False negatives

    # Precision and recall calculations
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# Create a DataFrame to hold the results
results_df = pd.DataFrame({
    'Class': class_names,
    'Precision': precisions,
    'Recall': recalls,
    'F1-Score': f1_scores
})

# Set the Class column as the index
results_df.set_index('Class', inplace=True)

# Print the results in table format
print(results_df)
