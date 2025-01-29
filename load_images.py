import os
import numpy as np
from PIL import Image

def augment_image(image):
    rotated_img = image.rotate(90)
    flipped_img = image.transpose(Image.FLIP_TOP_BOTTOM)
    return [image, rotated_img, flipped_img]

def load_and_augment_images(base_path):
    categorias = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for categoria in categorias:
        train_path = os.path.join(base_path, 'Training', categoria)
        test_path = os.path.join(base_path, 'Testing', categoria)
        class_num = categorias.index(categoria)

        for img in os.listdir(train_path):
            try:
                img_path = os.path.join(train_path, img)
                image = Image.open(img_path).convert('L')
                image = image.resize((80, 80))
                augmented_images = augment_image(image)
                for augmented_image in augmented_images:
                    train_data.append(np.array(augmented_image))
                    train_labels.append(class_num)
            except Exception as e:
                print(f"Erro ao carregar a imagem {img_path}: {e}")

        for img in os.listdir(test_path):
            try:
                img_path = os.path.join(test_path, img)
                image = Image.open(img_path).convert('L')
                image = image.resize((80, 80))
                augmented_images = augment_image(image)
                for augmented_image in augmented_images:
                    test_data.append(np.array(augmented_image))
                    test_labels.append(class_num)
            except Exception as e:
                print(f"Erro ao carregar a imagem {img_path}: {e}")

    return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)