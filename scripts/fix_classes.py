from pathlib import Path

train_dir = Path("data/processed/train")

for class_folder in train_dir.iterdir():
    if not class_folder.is_dir():
        continue
    old_name = class_folder.name            # e.g. "00012"
    new_name = str(int(old_name))           # turns "00012" → "12"
    new_folder = train_dir / new_name
    print(f"Renaming {class_folder} → {new_folder}")
    class_folder.rename(new_folder)
