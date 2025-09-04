import os
import shutil
from sklearn.model_selection import train_test_split

# Set your dataset path
original_dataset_dir = r"E:\Research\dataset"
output_base_dir = r"E:\Research\split_dataset"

# Paths for the categories
vehicle_dir = os.path.join(original_dataset_dir, "vehicles")
non_vehicle_dir = os.path.join(original_dataset_dir, "nonvehicles")

# Validate paths
if not os.path.exists(vehicle_dir):
    raise FileNotFoundError(f"Path not found: {vehicle_dir}")
if not os.path.exists(non_vehicle_dir):
    raise FileNotFoundError(f"Path not found: {non_vehicle_dir}")

# List images
vehicle_images = [os.path.join(vehicle_dir, img) for img in os.listdir(vehicle_dir)
                  if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

non_vehicle_images = [os.path.join(non_vehicle_dir, img) for img in os.listdir(non_vehicle_dir)
                      if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"Total vehicle images: {len(vehicle_images)}")
print(f"Total non-vehicle images: {len(non_vehicle_images)}")

# Split 80% train and 20% test
train_vehicles, test_vehicles = train_test_split(vehicle_images, test_size=0.2, random_state=42)
train_non_vehicles, test_non_vehicles = train_test_split(non_vehicle_images, test_size=0.2, random_state=42)

# Copy files with progress
def copy_files(file_list, destination_folder, label):
    os.makedirs(destination_folder, exist_ok=True)
    for i, file_path in enumerate(file_list, start=1):
        try:
            shutil.copy(file_path, destination_folder)
            if i % 500 == 0 or i == len(file_list):
                print(f"{label}: Copied {i}/{len(file_list)} files to {destination_folder}")
        except Exception as e:
            print(f"Error copying {file_path}: {e}")

# Folder structure and corresponding data
folders = [
    (train_vehicles, os.path.join(output_base_dir, "train", "vehicles"), "Train-Vehicles"),
    (test_vehicles, os.path.join(output_base_dir, "test", "vehicles"), "Test-Vehicles"),
    (train_non_vehicles, os.path.join(output_base_dir, "train", "non-vehicles"), "Train-NonVehicles"),
    (test_non_vehicles, os.path.join(output_base_dir, "test", "non-vehicles"), "Test-NonVehicles"),
]

# Execute copying
for file_list, dest_path, label in folders:
    copy_files(file_list, dest_path, label)

print("\n✅ Dataset split and copy completed successfully.")
