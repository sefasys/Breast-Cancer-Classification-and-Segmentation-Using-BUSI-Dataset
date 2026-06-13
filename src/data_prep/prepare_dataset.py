import os
import shutil
import random
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def combine_masks(mask_paths):
    """Combine multiple masks using bitwise OR."""
    if not mask_paths:
        return None
    
    # Read the first mask
    combined = np.array(Image.open(mask_paths[0]).convert('L'))
    
    # OR with subsequent masks
    for path in mask_paths[1:]:
        mask = np.array(Image.open(path).convert('L'))
        combined = np.bitwise_or(combined, mask)
        
    return Image.fromarray(combined)

def augment_pair(img, mask):
    """Apply the exact same random transformations to both image and mask."""
    angle = random.uniform(-15, 15)
    do_flip = random.choice([True, False])
    
    if do_flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        
    # Rotate (bicubic for image, nearest for mask to keep it binary)
    img = img.rotate(angle, resample=Image.BICUBIC)
    mask = mask.rotate(angle, resample=Image.NEAREST)
    
    return img, mask

def prepare_dataset(source_dir, dest_dir):
    categories = ['benign', 'malignant', 'normal']
    
    # Collect all items
    dataset_info = {cat: [] for cat in categories}
    
    for category in categories:
        cat_path = os.path.join(source_dir, category)
        if not os.path.exists(cat_path):
            continue
            
        for folder_id in os.listdir(cat_path):
            folder_path = os.path.join(cat_path, folder_id)
            if not os.path.isdir(folder_path):
                continue
                
            files = os.listdir(folder_path)
            masks = sorted([f for f in files if '_mask' in f])
            base_images = sorted([f for f in files if '_mask' not in f and f.endswith('.png')])
            
            if base_images:
                base_img = base_images[0]
                dataset_info[category].append({
                    'id': folder_id,
                    'base_img_name': base_img,
                    'base_img_path': os.path.join(folder_path, base_img),
                    'mask_paths': [os.path.join(folder_path, m) for m in masks]
                })

    # Perform stratified split (70/15/15)
    splits = {'train': {}, 'val': {}, 'test': {}}
    
    for category, items in dataset_info.items():
        if not items:
            continue
            
        # First split: 70% train, 30% temp
        train_items, temp_items = train_test_split(items, test_size=0.30, random_state=42)
        # Second split: 15% val, 15% test (which is 50% of the 30% temp)
        val_items, test_items = train_test_split(temp_items, test_size=0.50, random_state=42)
        
        splits['train'][category] = train_items
        splits['val'][category] = val_items
        splits['test'][category] = test_items
        
    # Create target directories and copy/combine files
    print("Creating splits and consolidating masks...")
    for split_name in ['train', 'val', 'test']:
        for category in categories:
            items = splits[split_name].get(category, [])
            for item in items:
                target_folder = os.path.join(dest_dir, split_name, category, item['id'])
                os.makedirs(target_folder, exist_ok=True)
                
                # Copy base image
                target_img_path = os.path.join(target_folder, item['base_img_name'])
                shutil.copy2(item['base_img_path'], target_img_path)
                
                # Combine and save mask
                if item['mask_paths']:
                    combined_mask = combine_masks(item['mask_paths'])
                    # Generate mask name based on base image name
                    mask_name = item['base_img_name'].replace('.png', '_mask.png')
                    target_mask_path = os.path.join(target_folder, mask_name)
                    combined_mask.save(target_mask_path)
                    
    # Class Balancing (Offline Augmentation) only on TRAIN set
    print("Performing offline class balancing on the TRAIN set...")
    train_counts = {cat: len(splits['train'].get(cat, [])) for cat in categories}
    
    # Assuming benign is always the majority, but let's find the actual max
    max_count = max(train_counts.values())
    
    for category in categories:
        current_count = train_counts[category]
        if current_count < max_count:
            needed = max_count - current_count
            print(f"Augmenting {category}: Needs {needed} more samples to reach {max_count}.")
            
            # Select random items from the existing TRAIN set to augment
            items_to_augment = splits['train'][category]
            
            for i in range(needed):
                # Pick a random item
                item = random.choice(items_to_augment)
                orig_id = item['id']
                new_id = f"aug_{orig_id}_{i}"
                
                # We need to read the already copied (and combined mask) files from train dir
                train_cat_dir = os.path.join(dest_dir, 'train', category)
                orig_folder = os.path.join(train_cat_dir, orig_id)
                new_folder = os.path.join(train_cat_dir, new_id)
                os.makedirs(new_folder, exist_ok=True)
                
                # Determine names
                base_img_name = item['base_img_name']
                mask_name = base_img_name.replace('.png', '_mask.png')
                
                # To match the new ID, we should rename the file. e.g. "benign (10).png" -> "benign (aug_10_0).png"
                # It's safer to just prepend "aug_" to the original name
                new_img_name = f"aug_{base_img_name}"
                new_mask_name = f"aug_{mask_name}"
                
                # Load images
                img_path = os.path.join(orig_folder, base_img_name)
                mask_path = os.path.join(orig_folder, mask_name)
                
                img = Image.open(img_path)
                mask = Image.open(mask_path)
                
                # Apply Augmentation
                aug_img, aug_mask = augment_pair(img, mask)
                
                # Save
                aug_img.save(os.path.join(new_folder, new_img_name))
                aug_mask.save(os.path.join(new_folder, new_mask_name))

    print(f"\nDataset Preparation Completed! Output directory: {dest_dir}")
    print("Train splits after balancing:")
    for cat in categories:
        final_train_path = os.path.join(dest_dir, 'train', cat)
        if os.path.exists(final_train_path):
            print(f" - {cat}: {len(os.listdir(final_train_path))} cases")

if __name__ == "__main__":
    SOURCE_DIR = "/home/sefasys/Desktop/Breast_Ultrasound_Dataset/Dataset_BUSI_with_GT"
    DEST_DIR = "/home/sefasys/Desktop/Breast_Ultrasound_Dataset/Dataset_BUSI_with_GT_Split"
    
    random.seed(42)
    prepare_dataset(SOURCE_DIR, DEST_DIR)
