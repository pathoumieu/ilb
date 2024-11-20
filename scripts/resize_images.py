import os
from PIL import Image
from icecream import ic
from pathlib import Path
from tqdm import tqdm
from utils_torch import resize_with_padding

TARGET_SIZE = 128


if __name__ == "__main__":

    train_folder = "./data/reduced_images_ILB/reduced_images/train/"
    train_out_folder = f"./data/reduced_images_ILB/processed_images_{TARGET_SIZE}/train"
    test_folder = "./data/reduced_images_ILB/reduced_images/test/"
    test_out_folder = f"./data/reduced_images_ILB/processed_images_{TARGET_SIZE}/test"

    for data_folder, out_folder in tqdm(zip([train_folder, test_folder], [train_out_folder, test_out_folder])):
        for announce_folder in tqdm(os.listdir(data_folder)):
            (Path(out_folder) / announce_folder).mkdir(exist_ok=True, parents=True)
            for img_path in os.listdir(Path(data_folder) / announce_folder):
                img = Image.open(Path(data_folder) / announce_folder / img_path)
                resized_img = resize_with_padding(img, target_size=TARGET_SIZE)
                resized_img.save(Path(out_folder) / announce_folder / img_path)
