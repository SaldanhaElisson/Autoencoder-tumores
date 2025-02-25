# 🧠 Brain Tumor Detection Using CNN [English version 🇺🇸]

🌍 Overview

This project aims to detect brain tumors using Convolutional Neural Networks (CNNs). The dataset consists of medical images, and the model is trained to efficiently classify different types of tumors.

📊 Dataset

- **Total Samples:** 9792  
- **Training Data:** 90% (8812 images)  
- **Test Data:** 10% (980 images)  
- **Image Type:** Medical MRI scans  

⚙️ Methodology

The methodology consists of three main phases: data preprocessing, model selection, and training & evaluation. During preprocessing, images are resized, normalized, and augmented when necessary to improve model generalization. The chosen architecture is a 2D CNN with multiple convolutional layers, followed by fully connected layers to classify images into different tumor types. The model is trained using the Adam optimizer, and performance is evaluated through metrics such as accuracy, precision, recall, and F1-score.

🏗️ **Architecture of 2D CNN**

The model receives grayscale images of 80x80 as input. It is structured into eight convolutional layers:  
- **First two layers:** 64 filters  
- **Next two layers:** 32 filters  
- **Following two layers:** 16 filters  
- **Final two layers:** 8 filters  
- Each convolutional layer employs a **2x2 kernel**, and pooling layers are strategically inserted after some convolutional layers.  
- **Batch normalization** is applied throughout the architecture to enhance feature extraction and reduce overfitting.  
- **Dropout rate:** 0.1 after pooling and dense layers.  
- The network ends with a **fully connected layer** (1024 neurons), followed by a **softmax layer** that classifies images into four categories.  

The model is trained using **Adam optimizer**, with learning rates of **0.01, 0.001, and 0.0001** tested, and **categorical cross-entropy** as the loss function. Training occurs over **100 epochs** with a **batch size of 16**.

📈 **Model Training**

- 🛠️ **Optimizer:** Adam  
- 🎯 **Loss Function:** Categorical Cross-Entropy  
- ⏳ **Training Time per Epoch:** ~7s  
- 🧮 **Total Trainable Parameters:** 243,924  
- 💻 **Hardware Used:** NVIDIA RTX 4060  

🏆 **Results**

- 📊 **Accuracy:** The model achieved high classification accuracy.  
- 📉 **Loss Reduction:** Training loss consistently decreased over epochs.  

🏗️ **Autoencoder Architecture**

The Autoencoder was implemented for **preprocessing and feature extraction** before the CNN model. It consists of **encoder-decoder convolutional layers**, with a **compressed code layer** for efficient representation.  
Training was performed using the **Adam optimizer** and the **MSE (Mean Squared Error) loss function**, ensuring that reconstructed images closely matched the originals.  

💻 **Hardware Used:** NVIDIA RTX 3060  

👨‍💻 **Contributors**  
- [João Pedro Lima](https://github.com/joaopedrolima)  
- [Elisson Saldanha](https://github.com/elissonsaldanha)  

📜 **License**  
MIT License  

---

# 🧠 Brain Tumor Detection Using CNN [Portuguese version 🇧🇷]

🌍 **Visão Geral**  

Este projeto tem como objetivo **detectar tumores cerebrais** usando Redes Neurais Convolucionais (CNNs). O conjunto de dados é composto por imagens médicas, e o modelo é treinado para classificar **diferentes tipos de tumores** de forma eficiente.  

📊 **Conjunto de Dados**  

- **Total de Amostras:** 9792  
- **Dados de Treinamento:** 90% (8812 imagens)  
- **Dados de Teste:** 10% (980 imagens)  
- **Tipo de Imagem:** Scans de ressonância magnética (MRI) médicas  

⚙️ **Metodologia**  

A metodologia consiste em três fases principais: **pré-processamento dos dados, seleção do modelo e treinamento & avaliação**. Durante o pré-processamento, as imagens são **redimensionadas, normalizadas e aumentadas** quando necessário para melhorar a generalização do modelo. A arquitetura escolhida é uma **CNN 2D** com múltiplas camadas convolucionais, seguidas por camadas totalmente conectadas para classificar as imagens em diferentes tipos de tumor. O treinamento do modelo utiliza o **otimizador Adam**, e o desempenho é avaliado através de métricas como **acurácia, precisão, recall e F1-score**.  

🏗️ **Arquitetura da CNN 2D**  

O modelo recebe imagens em **tons de cinza** de **80x80** como entrada. Ele é estruturado em **oito camadas convolucionais**:  
- **Duas primeiras camadas:** 64 filtros  
- **Duas seguintes:** 32 filtros  
- **Duas posteriores:** 16 filtros  
- **Duas últimas:** 8 filtros  
- Cada camada convolucional emprega um **kernel de 2x2**, e camadas de **pooling** são inseridas estrategicamente após algumas camadas convolucionais.  
- Para melhorar a **extração de características** e reduzir **overfitting**, **normalização em lote (batch normalization)** é aplicada em toda a arquitetura.  
- **Taxa de Dropout:** 0.1 após camadas de pooling e densas.  
- A rede culmina em uma **camada totalmente conectada** com **1024 neurônios**, seguida por uma **camada softmax** que classifica as imagens em **quatro categorias**.  

O modelo é treinado usando o **otimizador Adam**, com **taxas de aprendizado de 0.01, 0.001 e 0.0001** testadas, e a função de perda utilizada é **entropia cruzada categórica**. O treinamento ocorre por **100 épocas** com um **tamanho de lote de 16**.  

📈 **Treinamento do Modelo**  

- 🛠️ **Otimizador:** Adam  
- 🎯 **Função de Perda:** Entropia Cruzada Categórica  
- ⏳ **Tempo de Treinamento por Época:** ~7s  
- 🧮 **Parâmetros Treináveis Totais:** 243,924  
- 💻 **Hardware Utilizado:** NVIDIA RTX 4060  

🏆 **Resultados**  

- 📊 **Acurácia:** O modelo atingiu alta precisão na classificação.  
- 📉 **Redução da Perda:** A perda de treinamento diminuiu consistentemente ao longo das épocas.  

🏗️ **Arquitetura do Autoencoder**  

O **Autoencoder** foi implementado para **pré-processamento e extração de características** antes do modelo CNN. Ele consiste em **camadas convolucionais encoder-decoder**, com uma **camada de código comprimido** para representação eficiente.  
O treinamento foi realizado utilizando o **otimizador Adam** e a função de perda **MSE (Mean Squared Error)**, garantindo que as imagens reconstruídas fossem próximas às originais.  

💻 **Hardware Utilizado:** NVIDIA RTX 3060  

👨‍💻 **Contribuidores**  
- [João Pedro Lima](https://github.com/joaopedrolima)  
- [Elisson Saldanha](https://github.com/elissonsaldanha)  

📜 **Licença**  
MIT License  
