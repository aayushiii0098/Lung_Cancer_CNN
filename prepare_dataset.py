import os
import shutil
import random

dataset_path = "dataset"

train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")

classes = {
    "Benign cases": "cancer",
    "Malignant cases": "cancer",
    "Normal cases": "normal"
}

# Create folders 
for folder in [train_path, test_path]:
    for label in ["cancer", "normal"]:
        os.makedirs(os.path.join(folder, label), exist_ok=True)

for class_folder, label in classes.items():

    folder_path = os.path.join(dataset_path, class_folder)
    images = os.listdir(folder_path)

    random.shuffle(images)

    split = int(0.8 * len(images))

    train_images = images[:split]
    test_images = images[split:]

    for img in train_images:
        shutil.copy(
            os.path.join(folder_path, img),
            os.path.join(train_path, label, img)
        )

    for img in test_images:
        shutil.copy(
            os.path.join(folder_path, img),
            os.path.join(test_path, label, img)
        )

print("Dataset prepared successfully!")