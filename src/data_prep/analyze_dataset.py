import os
from collections import defaultdict
from PIL import Image

def analyze_dataset(base_dir):
    categories = ['benign', 'malignant', 'normal']
    stats = defaultdict(lambda: {
        'count': 0, 
        'masks_count': 0, 
        'multiple_masks': 0,
        'widths': [],
        'heights': []
    })
    
    total_images = 0
    
    for category in categories:
        category_path = os.path.join(base_dir, category)
        if not os.path.exists(category_path):
            continue
            
        for folder_id in os.listdir(category_path):
            folder_path = os.path.join(category_path, folder_id)
            if not os.path.isdir(folder_path):
                continue
                
            stats[category]['count'] += 1
            total_images += 1
            
            # Count images and masks
            files = os.listdir(folder_path)
            masks = [f for f in files if '_mask' in f]
            base_images = [f for f in files if '_mask' not in f and f.endswith('.png')]
            
            if len(masks) > 1:
                stats[category]['multiple_masks'] += 1
            stats[category]['masks_count'] += len(masks)
            
            for img_name in base_images:
                img_path = os.path.join(folder_path, img_name)
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                        stats[category]['widths'].append(w)
                        stats[category]['heights'].append(h)
                except Exception as e:
                    pass
    
    print("--- Dataset Analysis ---")
    print(f"Total base images (cases): {total_images}")
    for cat, data in stats.items():
        print(f"\nCategory: {cat.upper()}")
        print(f"  - Number of cases: {data['count']}")
        print(f"  - Total masks: {data['masks_count']}")
        print(f"  - Cases with multiple masks: {data['multiple_masks']}")
        
        if data['widths'] and data['heights']:
            avg_w = sum(data['widths']) / len(data['widths'])
            avg_h = sum(data['heights']) / len(data['heights'])
            print(f"  - Average dimensions: {avg_w:.1f} x {avg_h:.1f}")
            print(f"  - Min/Max width: {min(data['widths'])} / {max(data['widths'])}")
            print(f"  - Min/Max height: {min(data['heights'])} / {max(data['heights'])}")
        else:
            print("  - No image dimensions found (could not read images).")

if __name__ == "__main__":
    analyze_dataset("/home/sefasys/Desktop/Breast_Ultrasound_Dataset/Dataset_BUSI_with_GT")
