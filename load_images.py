import os
import numpy as np
from PIL import Image

def augment_image(image):
    rotated_img = image.rotate(90)
    flipped_img = image.transpose(Image.FLIP_TOP_BOTTOM)
    return [image, rotated_img, flipped_img]

def load_and_augment_images(base_path):
    categorias = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
    data = []
    labels = []

    for categoria in categorias:
        path = os.path.join(base_path, categoria)
        class_num = categorias.index(categoria)
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                image = Image.open(img_path).convert('L')
                image = image.resize((80, 80))
                augmented_images = augment_image(image)
                for augmented_image in augmented_images:
                    data.append(np.array(augmented_image))
                    labels.append(class_num)
            except Exception as e:
                print("Erro ao carregar a imagem {img_path}: {e}".format(img_path=img_path, e=e))

    return np.array(data), np.array(labels)
