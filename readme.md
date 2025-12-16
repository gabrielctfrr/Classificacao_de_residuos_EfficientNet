# Classificação de Resíduos com EfficientNet (PyTorch)

## 📌 Visão Geral

Este projeto tem como objetivo o desenvolvimento de um modelo de **Deep Learning para classificação de resíduos sólidos** a partir de imagens, utilizando **EfficientNet-B0** e **PyTorch**.

O modelo é capaz de classificar imagens em seis categorias:

* Cardboard (papelão)
* Glass (vidro)
* Metal (metal)
* Paper (papel)
* Plastic (plástico)
* Trash (lixo comum)

O projeto foi desenvolvido com foco em **boas práticas de visão computacional**, organização de código e preparação para uso em contexto profissional.

---

## 🧠 Tecnologias Utilizadas

* Python
* PyTorch
* Torchvision
* EfficientNet (transfer learning)
* OpenCV (exploração inicial)
* Jupyter Notebook

---

## 📂 Estrutura do Projeto

```
Classificacao_de_residuos_EfficientNet/
│
├── data/
│   ├── train/
│   │   ├── cardboard/
│   │   ├── glass/
│   │   ├── metal/
│   │   ├── paper/
│   │   ├── plastic/
│   │   └── trash/
│   └── val/
│       ├── cardboard/
│       ├── glass/
│       ├── metal/
│       ├── paper/
│       ├── plastic/
│       └── trash/
│
├── notebook/
│   └── exploration.ipynb
│
├── src/
│   ├── train.py
│   ├── model.py
│   ├── dataset.py
│   └── evaluate.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

> ⚠️ O dataset **não está incluído** neste repositório por questões de tamanho.

---

## 📊 Dataset

Foi utilizado um **dataset público de classificação de resíduos**, amplamente utilizado em projetos de visão computacional.

As imagens são organizadas por classe em pastas, compatíveis com o `ImageFolder` do Torchvision.

📎 Dataset disponível em:

* [https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)

---

## ⚙️ Metodologia

* Pré-processamento das imagens (resize, normalização e data augmentation)
* Uso de **transfer learning** com EfficientNet-B0 pré-treinada no ImageNet
* Congelamento do backbone e treinamento do classificador final
* Treinamento supervisionado com função de perda CrossEntropy
* Avaliação do modelo em conjunto de validação

---

## 📈 Resultados

O modelo alcançou aproximadamente:

* **76% de acurácia no conjunto de validação**

Resultado obtido sem fine-tuning avançado, demonstrando boa capacidade de generalização do modelo.

---

## ▶️ Como Executar o Projeto

### 1️⃣ Criar ambiente virtual (opcional)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 2️⃣ Instalar dependências

```bash
pip install -r requirements.txt
```

### 3️⃣ Organizar o dataset


### 4️⃣ Treinar o modelo

```bash
python src/train.py
```

---

## 🚀 Possíveis Melhorias Futuras

* Fine-tuning das camadas finais da EfficientNet
* Avaliação com matriz de confusão e métricas adicionais
* Otimização do modelo para inferência em tempo real
* Deploy do modelo como API

---

## 👤 Autor

Projeto desenvolvido por **Gabriel** como parte de estudos em **Visão Computacional e Deep Learning**, com foco em aplicações práticas e preparação para vagas de nível Júnior.

---

## 📄 Licença

Este projeto é apenas para fins educacionais.
