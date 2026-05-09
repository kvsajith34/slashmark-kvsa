import random
import shutil
from pathlib import Path

def split_data():
    # Create val folders
    Path('data/val/cats').mkdir(parents=True, exist_ok=True)
    Path('data/val/dogs').mkdir(parents=True, exist_ok=True)
    
    # Cats: 20% to val
    cat_files = list(Path('data/cats/images').glob('*.*'))
    n_cat_val = int(len(cat_files) * 0.2)
    selected_cats = random.sample(cat_files, n_cat_val)
    for f in selected_cats:
        shutil.move(str(f), f'data/val/cats/{f.name}')
    print(f"Moved {n_cat_val} cat images to val")
    
    # Dogs: 20% to val
    dog_files = list(Path('data/dogs/images').glob('*.*'))
    n_dog_val = int(len(dog_files) * 0.2)
    selected_dogs = random.sample(dog_files, n_dog_val)
    for f in selected_dogs:
        shutil.move(str(f), f'data/val/dogs/{f.name}')
    print(f"Moved {n_dog_val} dog images to val")
    
    print("\n✅ Split complete! 80% training / 20% validation")

split_data()