import os
import csv
import shutil
from pathlib import Path
from PIL import Image

# === CONFIGURATION ===
RAW_TRAIN_DIR = Path("data/raw/Final_Training/Images")
RAW_TEST_DIR  = Path("data/raw/Final_Test/Images")
TEST_CSV      = Path("data/raw/Final_Test/GT-final_test.csv")

PROC_TRAIN_DIR = Path("data/processed/train")
PROC_TEST_DIR  = Path("data/processed/test")

IMG_SIZE = (64, 64)  # (width, height)


def make_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def process_training():
    for class_folder in RAW_TRAIN_DIR.iterdir():
        if not class_folder.is_dir():
            continue
        class_id = class_folder.name
        out_dir = PROC_TRAIN_DIR / class_id
        make_dir(out_dir)

        for img_path in class_folder.glob("*.ppm"):
            with Image.open(img_path) as img:
                img = img.resize(IMG_SIZE)
                save_path = out_dir / (img_path.stem + ".png")
                img.save(save_path)


def process_test():
    # First read CSV mapping test images to class IDs
    mapping = {}
    with open(TEST_CSV, newline="") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=";")
        for row in reader:
            mapping[row["Filename"]] = row["ClassId"]

    # Then process each image
    for img_path in RAW_TEST_DIR.glob("*.ppm"):
        fname = img_path.name
        class_id = mapping.get(fname)
        if class_id is None:
            continue  # skip if not in CSV

        out_dir = PROC_TEST_DIR / class_id
        make_dir(out_dir)

        with Image.open(img_path) as img:
            img = img.resize(IMG_SIZE)
            save_path = out_dir / (img_path.stem + ".png")
            img.save(save_path)


if __name__ == "__main__":
    print("Processing training images...")
    process_training()
    print("Processing test images...")
    process_test()
    print("Done! Structured, resized images are in data/processed/")
