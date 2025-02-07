import numpy as np
from keras.src.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten, BatchNormalization, Dropout
from keras.src.models import Model
from keras.src.optimizers import Adam
from keras.src.utils import to_categorical
import os
from load_images import load_and_augment_images
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score

print("Carregando e aumentando as imagens...")
base_path = os.path.join(os.getcwd(), 'data_base')
x_train, y_train, x_test, y_test = load_and_augment_images(base_path)
print("Imagens carregadas e aumentadas com sucesso.")

print("Normalizando as imagens...")
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
print("Imagens normalizadas com sucesso.")

print("Redimensionando as imagens...")
x_train = x_train.reshape((x_train.shape[0], 80, 80, 1))
x_test = x_test.reshape((x_test.shape[0], 80, 80, 1))
print("Imagens redimensionadas com sucesso.")

print("Convertendo rótulos para categorias...")
categories = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
y_train = to_categorical(y_train, num_classes=len(categories))
y_test = to_categorical(y_test, num_classes=len(categories))
print("Rótulos convertidos com sucesso.")

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

print("Construindo o modelo do autoencoder...")
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
print("Modelo do autoencoder construído e compilado com sucesso.")


print("Construindo o modelo do classificador...")
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
print("Modelo do classificador construído e compilado com sucesso.")


x_combined = np.concatenate((x_train, x_test), axis=0)
y_combined = np.concatenate((y_train, y_test), axis=0)

k = 3
kf = KFold(n_splits=k, shuffle=True, random_state=42)
epochas_in = 10
batch_size_in = 16

precision_scores = []
recall_scores = []
f1_scores = []
autoencoder_histories = []
classifier_histories = []

for k_round, (train_index, val_index) in enumerate(kf.split(x_combined)):
    print(f"K round: {k_round + 1}")
    x_train_fold, x_val_fold = x_combined[train_index], x_combined[val_index]
    y_train_fold, y_val_fold = y_combined[train_index], y_combined[val_index]

    print("Treinando o autoencoder...")
    history_autoencoder = autoencoder.fit(
        x_train_fold, x_train_fold,
        epochs=epochas_in, batch_size=batch_size_in, shuffle=True,
        validation_data=(x_val_fold, x_val_fold)
    )

    autoencoder_histories.append(history_autoencoder)
    print("Autoencoder treinado com sucesso.")

    print("Treinando o classificador...")
    history_classifier = classifier.fit(
        encoder.predict(x_train_fold), y_train_fold,
        epochs=epochas_in, batch_size=batch_size_in, shuffle=True,
        validation_data=(encoder.predict(x_val_fold), y_val_fold)
    )
    classifier_histories.append(history_classifier)
    print("Classificador treinado com sucesso.")

    print("Avaliando o classificador...")
    y_pred = classifier.predict(encoder.predict(x_val_fold))
    y_pred_class = np.argmax(y_pred, axis=1)
    y_val_class = np.argmax(y_val_fold, axis=1)

    precision = precision_score(y_val_class, y_pred_class, average='macro')
    recall = recall_score(y_val_class, y_pred_class, average='macro')
    f1 = f1_score(y_val_class, y_pred_class, average='macro')

    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    print("Métricas:")
    print(f"Precision: {precision}, Recall: {recall}, F1-score: {f1}")


mean_precision = np.mean(precision_scores)
mean_recall = np.mean(recall_scores)
mean_f1 = np.mean(f1_scores)

print("\nMétricas finais:")
print(f"Precision média: {mean_precision}")
print(f"Recall médio: {mean_recall}")
print(f"F1-score médio: {mean_f1}")

metrics_path = os.path.join(os.getcwd(), 'metrics.txt')
with open(metrics_path, 'w') as f:
    for k_round, (precision, recall, f1) in enumerate(zip(precision_scores, recall_scores, f1_scores)):
        f.write(f"K round {k_round + 1}:\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1-score: {f1}\n\n")
    f.write(f"Precision média: {mean_precision}\n")
    f.write(f"Recall médio: {mean_recall}\n")
    f.write(f"F1-score médio: {mean_f1}\n")
print(f"Métricas salvas em {metrics_path}.")


mean_autoencoder_loss = np.mean([h.history['loss'] for h in autoencoder_histories], axis=0)
mean_autoencoder_val_loss = np.mean([h.history['val_loss'] for h in autoencoder_histories], axis=0)

mean_classifier_loss = np.mean([h.history['loss'] for h in classifier_histories], axis=0)
mean_classifier_val_loss = np.mean([h.history['val_loss'] for h in classifier_histories], axis=0)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(mean_autoencoder_loss, label='Loss Treinamento')
plt.plot(mean_autoencoder_val_loss, label='Loss Validação')
plt.title('Curva de Aprendizagem do Autoencoder')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(mean_classifier_loss, label='Loss Treinamento')
plt.plot(mean_classifier_val_loss, label='Loss Validação')
plt.title('Curva de Aprendizagem do Classificador')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()

plot_path = os.path.join(os.getcwd(), 'learning_curves.png')
plt.savefig(plot_path)
print(f"Curvas de aprendizagem salvas em {plot_path}.")
plt.show()
plt.close()
print("Curvas de aprendizagem plotadas com sucesso.")

