import math
import os
import subprocess

import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def combine_images(generated_images,gen_num_images=10): # generated_images는 4d tensor
    num_images = generated_images.shape[0] # 이미지의 개수
    new_width = gen_num_images # 가로의 길이는 10개 
    new_height = int(num_images / new_width)
    grid_shape = generated_images.shape[1:3] # 예를들면, (48,48)가 저장
    grid_image = np.zeros((new_height * grid_shape[0], new_width * grid_shape[1]), dtype=generated_images.dtype) # 전체 이미지 공간 생성
    for index, img in enumerate(generated_images): # 70장이들어올꺼임.
        i,j = divmod(index, new_width)
        grid_image[i * grid_shape[0]:(i + 1) * grid_shape[0], j * grid_shape[1]:(j + 1) * grid_shape[1]] = img[:, :, 0] # enumerate를 하면 나오는 img차원은 3d tensor(갯수부분이 사라짐)
    return grid_image


def generate_noise(shape: tuple): # 그냥 shape만큼 noise 생성.
    noise = np.random.randn(shape[0] ,shape[1]) #latent_dim * n_samples
    return noise


def generate_condition_embedding(label: int, nb_of_label_embeddings: int, fake_classes=100):
    label_embeddings = np.zeros((nb_of_label_embeddings, fake_classes)) # {갯수} x {class의 수}
    label_embeddings[:, label] = 1
    return label_embeddings # one-hot encdoing 완료


def generate_images(generator, nb_images: int, label: int , classes = 7): # 생성시킬 label과 이미지의 수를 입력
    noise = generate_noise((nb_images, 100))
    label_batch = generate_condition_embedding(label, nb_images , classes)
    generated_images = generator.predict([noise, label_batch], verbose=0)
    return generated_images


def generate_image_grid(generator, title: str = "Generated images",classes=7,gen_num_images=10,cmap=None):
    generated_images = []

    for i in range(classes): # 특정 class에 대해 아래의 행동 수행.
        noise = generate_noise((gen_num_images, 100))
        label_input = i* np.ones([gen_num_images],dtype=int) # 각 class의 label 10개 변환
        gen_images = generator.predict([noise, label_input], verbose=0) # 각 class의 이미지 10개 생성
        generated_images.extend(gen_images) # 이렇게되면, 70개의 이미지가 오른쪽으로 class묶음으로 생성

    generated_images = np.array(generated_images)
    image_grid = combine_images(generated_images) # 70개의 이미지를 전체 하나의 변수에 투영
    image_grid = inverse_transform_images(image_grid) # 이미지가 보이게끔 역변환

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    
	
    ax.imshow(image_grid, cmap=cmap)
    ax.set_title(title)
    fig.canvas.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()

    return image


def save_generated_image(image, epoch, iteration, folder_path): # 이미지 저장
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    file_path = "{0}/{1}_{2}.png".format(folder_path, epoch, iteration)
    cv2.imwrite(file_path, image.astype(np.uint8))


def transform_images(images: np.ndarray):  # 0~255를 -> -1~1로 변환
    """
    Transform images to [-1, 1]
    """

    images = (images.astype(np.float32) - 127.5) / 127.5
    return images


def inverse_transform_images(images: np.ndarray): # -1~1 -> 0~255로 변환
    """
    From the [-1, 1] range transform the images back to [0, 255]
    """

    images = images * 127.5 + 127.5
    images = images.astype(np.uint8)
    return images

