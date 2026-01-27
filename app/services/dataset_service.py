import random
from pathlib import Path
from ..core.config import get_dataset_path, get_v2_dataset_path

class DatasetCache:
    def __init__(self):
        self.v1_files = []
        self.v2_data = {} # { category_name: [file_names] }
        self.v2_root = None
        self.v1_root = None

    def refresh_v1(self):
        path_str = get_dataset_path()
        if not path_str:
            return
        
        self.v1_root = Path(path_str)
        if self.v1_root.exists():
            # Only store the names to save memory, full path can be reconstructed
            self.v1_files = [f.name for f in self.v1_root.glob("*.png")] + [f.name for f in self.v1_root.glob("*.jpg")]
            print(f"📦 V1 Cache Initialized: {len(self.v1_files)} files found.")

    def refresh_v2(self):
        path_str = get_v2_dataset_path()
        if not path_str:
            return
        
        self.v2_root = Path(path_str)
        if self.v2_root.exists():
            new_v2_data = {}
            # Exclude metadata/structural folders
            excluded = ["other", "labels", "images"]
            
            subdirs = [d for d in self.v2_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
            for d in subdirs:
                category = d.name
                files = [f.name for f in d.glob("*.png")] + [f.name for f in d.glob("*.jpg")]
                if files:
                    new_v2_data[category] = files
            
            self.v2_data = new_v2_data
            print(f"📦 V2 Cache Initialized: {len(self.v2_data)} categories found.")

    def get_random_v1(self):
        if not self.v1_files:
            self.refresh_v1()
        if not self.v1_files:
            return None
        
        filename = random.choice(self.v1_files)
        return self.v1_root / filename

    def get_v2_challenge(self):
        if not self.v2_data:
            self.refresh_v2()
        if not self.v2_data:
            return None

        # Filter out 'Other' for target category if possible
        available_categories = [c for c in self.v2_data.keys() if c.lower() != 'other']
        if not available_categories:
            available_categories = list(self.v2_data.keys())
            
        target_category = random.choice(available_categories)
        target_files = self.v2_data[target_category]
        
        num_target = random.randint(3, 5)
        selected_targets = random.sample(target_files, min(num_target, len(target_files)))
        
        grid_images = []
        for fname in selected_targets:
            grid_images.append({
                "id": fname,
                "category": target_category,
                "is_correct": True
            })
            
        # Fill distractors
        other_categories = [c for c in self.v2_data.keys() if c != target_category]
        num_distractors = 9 - len(grid_images)
        
        if other_categories:
            for _ in range(num_distractors):
                d_cat = random.choice(other_categories)
                d_fname = random.choice(self.v2_data[d_cat])
                grid_images.append({
                    "id": d_fname,
                    "category": d_cat,
                    "is_correct": False
                })
        
        random.shuffle(grid_images)
        return target_category, grid_images

# Global singleton instance
dataset_cache = DatasetCache()
