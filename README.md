# Final Summative Group Project
 Introduction to Artificial Intelligence Summative: Implementation and Testing

# Text-to-Image Generator using GANs

This project implements a Text-to-Image Generator using Generative Adversarial Networks (GANs). The application uses a deep learning model to generate images based on textual descriptions. It consists of a training script to train the GANs and a Flask web application to interact with the trained model.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)

## Overview

The project consists of a training script (`train.py`) that trains a GAN model to generate images from text descriptions, and a web application (`app.py`) that provides a user interface for generating images based on user-provided text descriptions.

## Features

- Train a GAN model on a custom dataset of text-image pairs.
- Generate images from text descriptions using a trained GAN model.
- Simple web interface for interacting with the model.

## Directory Structure

```
Project/
├── train.py
├── app.py
├── README.md
├── dataset/
│   ├── images/
│   └── text/
├── generated/
└── templates/
    └── index.html
```

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- transformers
- Flask
- PIL (Pillow)
- matplotlib

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kiki-etc/Final-Summative-Group-Project
   cd Final-Summative-Group-Project
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the pre-trained BERT tokenizer:
   ```bash
   python -c "from transformers import BertTokenizer; BertTokenizer.from_pretrained('bert-base-uncased')"
   ```

## Training the Model

1. Preparing the dataset:
   - Images were placed in the `dataset/images/` directory and the corresponding text descriptions in the `dataset/text/` directory.

2. The trained model weights will be saved as `generator.pth` and `discriminator.pth`.

## Running the Application

1. Ensure that the virtual environment is activated:
   ```bash
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. Run the Flask application:
   ```bash
   python app.py
   ```

3. Open your web browser and go to `http://127.0.0.1:5000/` to access the application.

## Usage

1. Enter a text description in the input field and click "Generate Images".
2. The application will display the generated images based on the provided description.

## Troubleshooting

- **ModuleNotFoundError:** Make sure all required packages are installed. Use `pip install -r requirements.txt` to install missing packages.
- **CUDA errors:** Ensure you have a compatible GPU and CUDA drivers installed. If not, use CPU by setting `device = torch.device('cpu')` in `train.py` and `app.py`.
- **FileNotFoundError:** Ensure the dataset paths are correct and the text-image pairs are properly aligned.

## Acknowledgements

This project uses pre-trained models from the Hugging Face Transformers library and is inspired by various GAN implementations in the deep learning community.