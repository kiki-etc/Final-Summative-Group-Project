import subprocess
import sys
import logging

# Import installed packages
from flask import Flask, request, jsonify, send_file, render_template
import torch
import torch.nn as nn
from torchvision.utils import save_image
import os
import json
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

text_dim = 4

latent_dim = 100

class Generator(nn.Module):
    def __init__(self, latent_dim, text_dim, img_channels):
        super(Generator, self).__init__()
        self.text_embedding = nn.Embedding(text_dim, latent_dim)
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(latent_dim * 2, 512, 4, 1, 0),
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
        text_embedding = self.text_embedding(text).sum(dim=1)
        x = torch.cat([noise, text_embedding], dim=1)
        x = x.unsqueeze(2).unsqueeze(3)
        return self.gen(x)

generator = Generator(latent_dim, text_dim, 3)
# generator.load_state_dict(torch.load('generator.pth', map_location=device))
# generator.to(device)
# generator.eval()

PROJECT_ID = "intro-to-ai-431719"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

vertexai.init(project=PROJECT_ID, location=LOCATION)
generation_model = ImageGenerationModel.from_pretrained("imagegeneration@006")

def generate_images(description, num_images=3):
    prompt = f"2D african art {description}"
    response = generation_model.generate_images(
        prompt=prompt,
        number_of_images=num_images,
        seed=42,
        add_watermark=False,
    )
    
    os.makedirs('generated', exist_ok=True)
    file_paths = []
    for i, image in enumerate(response.images):
        file_path = f'generated/{description.replace(" ", "_")}_{i}.png'
        with open(file_path, 'wb') as f:
            f.write(image)
        file_paths.append(file_path)
    return file_paths

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        description = data['description']
        images = generate_images(description)
        return jsonify(images=images)
    except Exception as e:
        logging.error(f"Error in /generate: {e}")
        return jsonify(error=str(e)), 500

@app.route('/images/<path:filename>')
def serve_image(filename):
    try:
        return send_file(os.path.join('generated', filename), mimetype='image/png')
    except Exception as e:
        logging.error(f"Error serving image {filename}: {e}")
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    logging.info("Starting Flask app...")
    app.run(debug=True)