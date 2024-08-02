# Uncomment and run the appropriate command for your operating system, if required
# No installation is reqiured on Google Colab / Kaggle notebooks

# Linux / Binder / Windows (No GPU)
# !pip install numpy matplotlib torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# Linux / Windows (GPU)
# pip install numpy matplotlib torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
 
# MacOS (NO GPU)
# !pip install numpy matplotlib torch torchvision torchaudio

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = ['numpy', 'flask', 'matplotlib', 'torch', 'torchvision', 'torchaudio', 'flask', 'transformers', 'os']

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        install(package)

from flask import Flask, request, send_file, render_template

import torch
import torch.nn as nn

from torchvision.utils import save_image
from transformers import BertTokenizer
import os

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models
latent_dim = 100
text_dim = 768  # BERT base model output dimension

class Generator(nn.Module):
    def __init__(self, latent_dim, text_dim, img_channels):
        super(Generator, self).__init__()
        self.text_embedding = nn.Linear(text_dim, latent_dim)
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, noise, text):
        text_embedding = self.text_embedding(text)
        x = torch.cat([noise, text_embedding], dim=1)
        x = x.unsqueeze(2).unsqueeze(3)
        return self.gen(x)

generator = Generator(latent_dim, text_dim, 3)
generator.load_state_dict(torch.load('generator.pth', map_location=device))
generator.to(device)
generator.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def generate_images(description, num_images=3):
    tokens = tokenizer(description, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    text_ids = tokens['input_ids'].to(device)
    noise = torch.randn(num_images, 100).to(device)
    with torch.no_grad():
        images = generator(noise, text_ids.repeat(num_images, 1))
    os.makedirs('generated', exist_ok=True)
    file_paths = []
    for i in range(num_images):
        file_path = f'generated/{description.replace(" ", "_")}_{i}.png'
        save_image(images[i], file_path, normalize=True)
        file_paths.append(file_path)
    return file_paths

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    description = request.form['description']
    images = generate_images(description)
    return render_template('index.html', images=images, description=description)

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_file(os.path.join('generated', filename), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
