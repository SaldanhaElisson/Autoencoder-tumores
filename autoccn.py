import numpy as np
import matplotlib.pyplot as plt
from keras.src.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten, BatchNormalization, Dropout
from keras.src.models import Model
from keras.src.optimizers import Adam
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import os

from  load_images import load_and_augment_images

base_path = os.path.join(os.getcwd(), 'data_base')
data, labels = load_and_augment_images(base_path)
data = data.astype('float32') / 255.0  # Normalize os dados
data = data.reshape((data.shape[0], 80, 80, 1))
categories = ["gilioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
labels_one_hot = to_categorical(labels, num_classes=len(categories))
x_train, x_test, y_train, y_test = train_test_split(data, labels_one_hot, test_size=0.1, random_state=42)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
input_img = Input(shape=(80, 80, 1))

x = Conv2D(32, (2, 2), activation='relu', padding='same')(input_img)
x = BatchNormalization()(x)
x = Conv2D(128, (2, 2), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (2, 2), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Dropout(0.1)(x)

x = Conv2D(64, (2, 2), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (2, 2), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Dropout(0.1)(x)

x = Conv2D(64, (2, 2), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(128, (2, 2), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (2, 2), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)

x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
x = BatchNormalization()(x)

decoded = Conv2D(1, (2, 2), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')

history_autoencoder = autoencoder.fit(x_train, x_train, epochs=100, batch_size=16, shuffle=True, validation_data=(x_test, x_test))

encoder = Model(inputs=input_img, outputs=autoencoder.get_layer('conv2d_5').output)

encoded_input = Input(shape=(20, 20, 64))
x = Conv2D(64, (2, 2), activation='relu', padding='same')(encoded_input)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Dropout(0.1)(x)

x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(4, activation='softmax')(x)

classifier = Model(encoded_input, x)

classifier.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

history_classifier = classifier.fit(encoder.predict(x_train), y_train, epochs=100, batch_size=16, shuffle=True, validation_data=(encoder.predict(x_test), y_test))

y_pred = classifier.predict(encoder.predict(x_test))
y_pred_class = np.argmax(y_pred, axis=1)

y_test_class = np.argmax(y_test, axis=1)

precision = precision_score(y_test_class, y_pred_class, average='macro')
recall = recall_score(y_test_class, y_pred_class, average='macro')
sensitividade = recall_score(y_test_class, y_pred_class, average='macro')
f_measure = f1_score(y_test_class, y_pred_class, average='macro')

print("Precision:", precision)
print("Recall:", recall)
print("Sensitividade:", sensitividade)
print("F-measure:", f_measure)


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history_autoencoder.history['loss'], label='Loss Treinamento')
plt.plot(history_autoencoder.history['val_loss'], label='Loss Validação')
plt.title('Curva de Aprendizagem do Autoencoder')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_classifier.history['loss'], label='Loss Treinamento')
plt.plot(history_classifier.history['val_loss'], label='Loss Validação')
plt.title('Curva de Aprendizagem do Classificador')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
