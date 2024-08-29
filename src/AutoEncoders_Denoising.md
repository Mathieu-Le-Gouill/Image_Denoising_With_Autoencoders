python
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np


image_size=1024
coef_down_sizing = image_size/2040.0

crop_width = 256
crop_height = 256

assert image_size % crop_width == 0

dataset_path = "C:/Users/fourn/Downloads/autoencoders/DIV2K/DIV2K/"

def show_img_with_matplotlib(color_img, title, pos, figsize=(10, 7)):
    """Shows an image using matplotlib capabilities"""
    plt.figure(figsize=figsize)  # Ajuste la taille de la figure pour une grande image
    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')
    plt.show()  # Affiche chaque image dans sa propre figure
    
# Liste pour stocker les images chargées


def load_images_from_dir(dossier_images) :
    images = []
    # Lister tous les fichiers dans le dossier
    fichiers = os.listdir(dossier_images)

    fichiers_images = [f for f in fichiers]

    # Charger chaque image et l'ajouter à la liste des images
    for i, fichier in enumerate(fichiers_images):
        chemin_complet = os.path.join(dossier_images, fichier)
        image = cv2.imread(chemin_complet)
        if image is not None:
            images.append(image)
            
    return images

y_train = load_images_from_dir(dataset_path + "DIV2K_train_HR")
y_valid = load_images_from_dir(dataset_path + "DIV2K_valid_HR")

print(len(y_train)+len(y_valid),"images loaded.")


python
def preprocess_images(images):
    preprocessed_images = []

    for i, img in enumerate(images):
        height, width, _ = img.shape

        # Redimensionnement de l'image en conservant les proportions d'origine
        resized_img = cv2.resize(img, (int(width * coef_down_sizing), int(height * coef_down_sizing)), interpolation=cv2.INTER_LINEAR)
            
        preprocessed_images.append(resized_img)

    return preprocessed_images

y_train = preprocess_images(y_train)
y_valid = preprocess_images(y_valid)

print("Number traning images :",len(y_train))
print("Number validation images :",len(y_valid))



python
def add_gaussian_noise_and_resize(images, scale_factor_range=(0.25, 0.45), noise_std_range=(3.5, 10.0)):
    noisy_images = []
    
    for i, img in enumerate(images):
        
        scale_factor = np.random.uniform(*scale_factor_range)
        noise_std = np.random.uniform(*noise_std_range)
        
        # Réduction de la taille de l'image
        width = int(img.shape[1] * scale_factor)
        height = int(img.shape[0] * scale_factor)
        dimensions = (width, height)
        reduced_image = cv2.resize(img, dimensions, interpolation=cv2.INTER_LINEAR)

        # Réagrandissement de l'image à sa taille originale
        resized_image = cv2.resize(reduced_image, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Ajout d'un bruit gaussien aléatoire
        noise_r = np.random.normal(scale=noise_std, size=img.shape[:2])
        noise_g = np.random.normal(scale=noise_std, size=img.shape[:2])
        noise_b = np.random.normal(scale=noise_std, size=img.shape[:2])

        # Fusionner les bruits dans une seule matrice de bruit
        noise = np.stack((noise_r, noise_g, noise_b), axis=-1)

        # Appliquer le bruit
        noisy_img = resized_image + noise
        
        # Clipper les valeurs pour s'assurer qu'elles sont dans la plage valide [0, 255]
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
        
        noisy_images.append(noisy_img) 
        
    return noisy_images


x_train = add_gaussian_noise_and_resize(y_train)
x_valid = add_gaussian_noise_and_resize(y_valid)

print("Number traning images :",len(x_train))
print("Number validation images :",len(x_valid))



python
import numpy as np

def crop_images(x, y, crop_width=crop_width, crop_height=crop_height):
    x_crops = []
    y_crops = []
    print(x[0].shape)
    for i in range(len(x)):
        for h in range(0,x[i].shape[0] // crop_height * crop_height,crop_height):
            for w in range(0,x[i].shape[1] // crop_width * crop_width, crop_width):
                x_crops.append(x[i][h:h+crop_height, w:w+crop_width, :])
                y_crops.append(y[i][h:h+crop_height, w:w+crop_width, :])
    
    return np.stack(x_crops), np.stack(y_crops)

# Découper les images
x_train_crops, y_train_crops = crop_images(x_train, y_train)
x_valid_crops, y_valid_crops = crop_images(x_valid, y_valid)

# Afficher les dimensions des données découpées
print("Train data shapes:", x_train_crops.shape, y_train_crops.shape)
print("Validation data shapes:", x_valid_crops.shape, y_valid_crops.shape)

for i in range(30,34):
    show_img_with_matplotlib(x_train_crops[i], f"x cropped image {i}", i)
    show_img_with_matplotlib(y_train_crops[i], f"y cropped image {i}", i)



python
x_train_crops = x_train_crops / 255.0
y_train_crops = y_train_crops / 255.0

x_valid_crops = x_valid_crops / 255.0
y_valid_crops = y_valid_crops / 255.0



python
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, concatenate
from tensorflow.keras.optimizers import Adam
# Encodeur
encoder_input = tf.keras.layers.Input(shape=(crop_width, crop_height, 3))

conv1 = Conv2D(64, (3, 3), padding='same', activation='relu')(encoder_input)
conv1 = BatchNormalization()(conv1)
conv1 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv1)
conv1 = BatchNormalization()(conv1)

pool1 = MaxPooling2D((2, 2), padding='same')(conv1)

conv2 = Conv2D(32, (3, 3), padding='same', activation='relu')(pool1)
conv2 = BatchNormalization()(conv2)
conv2 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv2)
conv2 = BatchNormalization()(conv2)

pool2 = MaxPooling2D((2, 2), padding='same')(conv2)

conv3 = Conv2D(16, (3, 3), padding='same', activation='relu')(pool2)
conv3 = BatchNormalization()(conv3)
conv3 = Conv2D(16, (3, 3), padding='same', activation='relu')(conv3)
conv3 = BatchNormalization()(conv3)

pool3 = MaxPooling2D((2, 2), padding='same')(conv3)

conv4 = Conv2D(8, (3, 3), padding='same', activation='relu')(pool3)
conv4 = BatchNormalization()(conv4)
conv4 = Conv2D(8, (3, 3), padding='same', activation='relu')(conv4)
conv4 = BatchNormalization()(conv4)

# Décodeur
up1 = UpSampling2D((2, 2))(conv4)

concat1 = concatenate([conv3, up1], axis=-1)
conv5 = Conv2D(16, (3, 3), padding='same', activation='relu')(concat1)
conv5 = BatchNormalization()(conv5)
conv5 = Conv2D(16, (3, 3), padding='same', activation='relu')(conv5)
conv5 = BatchNormalization()(conv5)

up2 = UpSampling2D((2, 2))(conv5)

concat2 = concatenate([conv2, up2], axis=-1)
conv6 = Conv2D(32, (3, 3), padding='same', activation='relu')(concat2)
conv6 = BatchNormalization()(conv6)
conv6 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv6)
conv6 = BatchNormalization()(conv6)

up3 = UpSampling2D((2, 2))(conv6)

concat3 = concatenate([conv1, up3], axis=-1)
conv7 = Conv2D(64, (3, 3), padding='same', activation='relu')(concat3)
conv7 = BatchNormalization()(conv7)
conv7 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv7)
conv7 = BatchNormalization()(conv7)

output = Conv2D(3, (1, 1), activation='sigmoid', padding='same')(conv7)

# Création du modèle
model = tf.keras.models.Model(inputs=encoder_input, outputs=output)

def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.9)

model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[psnr])



python
import numpy as np
import tensorflow as tf

def data_generator(x, y, batch_size):
    num_samples = len(x)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        yield x[batch_indices], y[batch_indices]

# Définir la taille du batch
batch_size = int(image_size/crop_width)

# Créer des datasets TensorFlow
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(x_train_crops, y_train_crops, batch_size), 
    output_types=(tf.float32, tf.float32), 
    output_shapes=(
        tf.TensorShape([batch_size, crop_width, crop_height, 3]), 
        tf.TensorShape([batch_size, crop_width, crop_height, 3])
    )
)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

valid_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(x_valid_crops, y_valid_crops, batch_size), 
    output_types=(tf.float32, tf.float32), 
    output_shapes=(
        tf.TensorShape([batch_size, crop_width, crop_height, 3]), 
        tf.TensorShape([batch_size, crop_width, crop_height, 3])
    )
)
valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)




python
# Define learning rate schedule
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 60:
        lr *= 1e-3
    elif epoch > 40:
        lr *= 1e-2
    elif epoch > 20:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

# Define learning rate scheduler
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

# Entraîner le modèle avec les images bruitées comme entrées et les images originales comme labels
history = model.fit(train_dataset,
                    epochs=80,
                    shuffle=False,
                    validation_data=valid_dataset,
                    callbacks=[lr_scheduler])  # Utilisation du jeu de validation

# Test du modèle
decoded_imgs = model.predict(x_valid_crops)

model.save('model_auto_encoder_image_denoising')



python


