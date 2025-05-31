import os
import shutil

def create_data_structure(base_dir, class_names):
    """
    Create the data directory structure for binary classification.
    base_dir: base directory where data_binary will be created
    class_names: list of class folder names
    """
    data_binary_dir = os.path.join(base_dir, 'data_binary')
    if not os.path.exists(data_binary_dir):
        os.makedirs(data_binary_dir)
        print(f"Created base data directory: {data_binary_dir}")
    else:
        print(f"Base data directory already exists: {data_binary_dir}")

    for class_name in class_names:
        class_dir = os.path.join(data_binary_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
            print(f"Created class directory: {class_dir}")
        else:
            print(f"Class directory already exists: {class_dir}")

def copy_sample_images(src_dir, dest_dir, max_images_per_class=10):
    """
    Copy sample images from src_dir to dest_dir for each class folder.
    This is a helper function to prepare minimal data for testing.
    """
    if not os.path.exists(src_dir):
        print(f"Source directory {src_dir} does not exist. Cannot copy images.")
        return

    for class_name in os.listdir(src_dir):
        src_class_dir = os.path.join(src_dir, class_name)
        dest_class_dir = os.path.join(dest_dir, class_name)
        if os.path.isdir(src_class_dir) and os.path.exists(dest_class_dir):
            images_copied = 0
            for file in os.listdir(src_class_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')):
                    src_file = os.path.join(src_class_dir, file)
                    dest_file = os.path.join(dest_class_dir, file)
                    shutil.copy2(src_file, dest_file)
                    images_copied += 1
                    if images_copied >= max_images_per_class:
                        break
            print(f"Copied {images_copied} images to {dest_class_dir}")

if __name__ == "__main__":
    # Define base directory as project root
    base_dir = os.path.abspath(os.path.dirname(__file__))

    # Define your class names here (example)
    class_names = ['cancer', 'no_cancer']

    # Create data directory structure
    create_data_structure(base_dir, class_names)

    # Optionally, copy sample images from an existing data source to the new structure
    # Uncomment and set src_dir if you have source images to copy for testing
    # src_dir = os.path.join(base_dir, 'data', 'original_images')
    # copy_sample_images(src_dir, os.path.join(base_dir, 'data_binary'))

    print("Data preparation complete. Please add your images to the respective class folders in 'data_binary'.")
