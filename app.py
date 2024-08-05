from flask import Flask, request, jsonify, send_file, render_template
import torch
import torch.nn as nn
from torchvision.utils import save_image
import os
import json

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer
with open('tokenizer.json', 'r') as f:
    word_to_idx = json.load(f)
text_dim = len(word_to_idx)

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
generator.load_state_dict(torch.load('generator.pth', map_location=device))
generator.to(device)
generator.eval()

def generate_images(description, num_images=3):
    tokens = [word_to_idx[word] for word in description.lower().split(', ') if word in word_to_idx]
    text_ids = torch.tensor(tokens).to(device).unsqueeze(0)
    noise = torch.randn(num_images, latent_dim).to(device)
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
    data = request.get_json()
    description = data['description']
    images = generate_images(description)
    return jsonify(images=images)

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_file(os.path.join('generated', filename), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)