import os
import shutil

def create_data_binary_structure(base_dir):
    data_binary_dir = os.path.join(base_dir, 'data_binary')
    cancer_dir = os.path.join(data_binary_dir, 'cancer')
    no_cancer_dir = os.path.join(data_binary_dir, 'no_cancer')

    os.makedirs(cancer_dir, exist_ok=True)
    os.makedirs(no_cancer_dir, exist_ok=True)

    print(f"Created directories: {cancer_dir}, {no_cancer_dir}")
    return cancer_dir, no_cancer_dir

def copy_images_to_binary(src_dirs_map, cancer_dir, no_cancer_dir):
    """
    src_dirs_map: dict mapping source folder paths to 'cancer' or 'no_cancer'
    """
    for src_dir, label in src_dirs_map.items():
        if not os.path.exists(src_dir):
            print(f"Source directory does not exist: {src_dir}")
            continue
        dest_dir = cancer_dir if label == 'cancer' else no_cancer_dir
        count = 0
        for root, _, files in os.walk(src_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')):
                    src_file = os.path.join(root, file)
                    dest_file = os.path.join(dest_dir, file)
                    shutil.copy2(src_file, dest_file)
                    count += 1
        print(f"Copied {count} images from {src_dir} to {dest_dir}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.dirname(__file__))

    # Map your existing folders to 'cancer' or 'no_cancer'
    # Please adjust the folder names if needed
    src_dirs_map = {
        os.path.join(base_dir, 'data', 'im_Dyskeratotic'): 'cancer',
        os.path.join(base_dir, 'data', 'im_Koilocytotic'): 'cancer',
        os.path.join(base_dir, 'data', 'im_Metaplastic'): 'cancer',
        os.path.join(base_dir, 'data', 'im_Parabasal'): 'no_cancer',
        os.path.join(base_dir, 'data', 'im_Superficial-Intermediate'): 'no_cancer',
    }

    cancer_dir, no_cancer_dir = create_data_binary_structure(base_dir)
    copy_images_to_binary(src_dirs_map, cancer_dir, no_cancer_dir)

    print("Data binary preparation complete. You can now run train_model.py.")
