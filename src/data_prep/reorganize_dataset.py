import os
import re
import shutil

def reorganize_dataset(base_dir):
    categories = ['benign', 'malignant', 'normal']
    
    for category in categories:
        category_path = os.path.join(base_dir, category)
        if not os.path.exists(category_path):
            print(f"Directory not found, skipping: {category_path}")
            continue
            
        print(f"Processing category: {category}...")
        
        # List all files in the directory
        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)
            
            # Skip if it's already a directory
            if os.path.isdir(file_path):
                continue
                
            # Regex to match the id. Example matches: 
            # "benign (100).png" -> id: "100"
            # "benign (100)_mask.png" -> id: "100"
            # "benign (100)_mask_1.png" -> id: "100"
            match = re.match(r'^([a-zA-Z]+) \((\d+)\)(?:_mask.*)?\.png$', filename)
            
            if match:
                file_id = match.group(2)
                
                # Create the target directory for this id (e.g., .../benign/100)
                id_folder_path = os.path.join(category_path, file_id)
                os.makedirs(id_folder_path, exist_ok=True)
                
                # Move the file into the new directory
                target_path = os.path.join(id_folder_path, filename)
                shutil.move(file_path, target_path)
                print(f"Moved: {filename} -> {category}/{file_id}/")
            else:
                print(f"Unmatched file format, skipped: {filename}")

if __name__ == "__main__":
    dataset_dir = "/home/sefasys/Desktop/Breast_Ultrasound_Dataset/Dataset_BUSI_with_GT"
    
    if os.path.exists(dataset_dir):
        reorganize_dataset(dataset_dir)
        print("Reorganization completed successfully!")
    else:
        print(f"Error: Dataset directory not found at {dataset_dir}")
