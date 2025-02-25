🧠 Brain Tumor Detection Using CNN

🌍 Overview

Este projeto tem como objetivo detectar tumores cerebrais usando Redes Neurais Convolucionais (CNNs). O conjunto de dados é composto por imagens médicas, e o modelo é treinado para classificar diferentes tipos de tumores de forma eficiente.

📊 Dataset

- Total de Amostras: 9792

- Dados de Treinamento: 90% (8812 imagens)

- Dados de Teste: 10% (980 imagens)

- Tipo de Imagem: Scans de ressonância magnética (MRI) médicas

⚙️ Methodology

A metodologia consiste em três fases principais: pré-processamento dos dados, seleção do modelo e treinamento & avaliação. Durante o pré-processamento, as imagens são redimensionadas, normalizadas e aumentadas quando necessário para melhorar a generalização do modelo. A arquitetura escolhida é uma CNN 2D com múltiplas camadas convolucionais, seguidas por camadas totalmente conectadas para classificar as imagens em diferentes tipos de tumor. O treinamento do modelo utiliza o otimizador Adam, e o desempenho é avaliado através de métricas como acurácia, precisão, recall e F1-score.

🏗️ Architecture of 2D CNN

O modelo recebe imagens em tons de cinza de 80x80 como entrada. Ele é estruturado em oito camadas convolucionais: as duas primeiras utilizam 64 filtros, as duas seguintes 32 filtros, depois duas camadas com 16 filtros, e as últimas duas com 8 filtros. Cada camada convolucional emprega um kernel de 2x2, e camadas de pooling são inseridas estrategicamente após algumas camadas convolucionais. Para melhorar a extração de características e reduzir o overfitting, a normalização em lote (batch normalization) é aplicada em toda a arquitetura, e uma taxa de dropout de 0.1 é definida após as camadas de pooling e densas. A rede culmina em uma camada totalmente conectada com 1024 neurônios, seguida por uma camada softmax que classifica as imagens em quatro categorias. O modelo é treinado usando o otimizador Adam, com diferentes taxas de aprendizado testadas (0.01, 0.001 e 0.0001), e a função de perda utilizada é a entropia cruzada categórica. O treinamento ocorre por 100 épocas com um tamanho de lote de 16.

📈 Model Training

🛠️ Otimizador: Adam

🎯 Função de Perda: Entropia Cruzada Categórica

⏳ Tempo de Treinamento por Época: ~7s

🧮 Parâmetros Treináveis Totais: 243,924

💻 Hardware Utilizado: NVIDIA RTX 4060

🏆 Results

📊 Acurácia: O modelo atingiu alta precisão na classificação

📉 Redução da Perda: A perda de treinamento diminuiu consistentemente ao longo das épocas

🏗️ Autoencoder Architecture

O Autoencoder foi implementado para pré-processamento e extração de características antes do modelo CNN. Ele consiste em camadas convolucionais encoder-decoder, com uma camada de código comprimido para representação eficiente. O treinamento foi realizado utilizando o otimizador Adam e a função de perda MSE (Mean Squared Error), garantindo que as imagens reconstruídas fossem próximas às originais.

💻 Hardware Utilizado: NVIDIA RTX 3060

👨‍💻 Contributors

- [João Pedro Lima](https://github.com/JpLimags)  
- [Elisson Saldanha](https://github.com/SaldanhaElisson)



