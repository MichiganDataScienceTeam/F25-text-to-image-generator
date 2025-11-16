import streamlit as st
import torch
from torch import nn
from torch.nn import functional as F
import clip
from PIL import Image
import os

st.set_page_config(
    page_title="Face Generation with CVAE",
    layout="centered"
)

@st.cache_resource
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()

# Model architecture from cvae_celeba_solution.ipynb
class CelebaCVAE(nn.Module):
    def __init__(self, image_channels, init_channels, latent_size, class_size, image_size=64):
        super(CelebaCVAE, self).__init__()
        self.image_channels = image_channels
        self.latent_size = latent_size
        self.class_size = class_size
        self.init_channels = init_channels
        self.image_size = image_size
        
        conv_output_size = init_channels * 8
        
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, init_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(),
            
            nn.Conv2d(init_channels, init_channels*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(init_channels*2),
            nn.ReLU(),
            
            nn.Conv2d(init_channels*2, init_channels*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(init_channels*4),
            nn.ReLU(),
            
            nn.Conv2d(init_channels*4, init_channels*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(init_channels*8),
            nn.ReLU(),
            
            nn.Conv2d(init_channels*8, init_channels*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(init_channels*8),
            nn.ReLU(),
            
            nn.Conv2d(init_channels*8, conv_output_size, kernel_size=2, stride=1, padding=0),
            nn.ReLU()
        )
        
        self.fc1 = nn.Linear(conv_output_size + self.class_size, 512)
        self.fc_mu = nn.Linear(512, self.latent_size)
        self.fc_logvar = nn.Linear(512, self.latent_size)
        self.fc2 = nn.Linear(self.latent_size + self.class_size, conv_output_size)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(conv_output_size, init_channels*8, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(init_channels*8),
            nn.ReLU(),
            
            nn.ConvTranspose2d(init_channels*8, init_channels*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(init_channels*8),
            nn.ReLU(),
            
            nn.ConvTranspose2d(init_channels*8, init_channels*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(init_channels*4),
            nn.ReLU(),
            
            nn.ConvTranspose2d(init_channels*4, init_channels*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(init_channels*2),
            nn.ReLU(),
            
            nn.ConvTranspose2d(init_channels*2, init_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(),
            
            nn.ConvTranspose2d(init_channels, self.image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x, c):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        inputs = torch.cat([h, c], 1)
        h_fc = F.relu(self.fc1(inputs))
        mu = self.fc_mu(h_fc)
        logvar = self.fc_logvar(h_fc)
        return mu, logvar
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sample = mu + eps * std
        return sample
    
    def decode(self, z, c):
        inputs = torch.cat([z, c], 1)
        h = F.relu(self.fc2(inputs))
        h = h.view(-1, self.init_channels * 8, 1, 1)
        return self.decoder(h)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar


# Load models
@st.cache_resource
def load_models():
    # Hyperparameters
    # TODO: change parameters to the ones used in the training
    latent_size = 128 
    clip_dim = 512
    init_channels = 64 
    image_size = 64
    image_channels = 3
    
    model = CelebaCVAE(image_channels, init_channels, latent_size, clip_dim, image_size).to(device)
    # TODO: make sure this path is correct
    model_path = 'celeba_cvae_model.pth'
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None, None
    
    st.info(f"Loading model from: {model_path}")
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    clip_model, _ = clip.load("ViT-B/32", device=device)
    
    return model, clip_model

# Same as in cvae_celeba_solution.ipynb
def generate_faces(model, clip_model, text_prompt, num_samples=1, temperature=1.0):    
    text = clip.tokenize([text_prompt]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text)
    
    with torch.no_grad():
        # TODO: Use your latent_size value for the second parameter of torch.randn
        sample_z = torch.randn(num_samples, 128).to(device) * temperature
        text_condition = text_features.repeat(num_samples, 1)
        samples = model.decode(sample_z, text_condition).cpu()
    
    return samples

# Used to transform the tensor to a PIL image that can be displayed through streamlit
def tensor_to_pil_image(tensor):
    """Convert tensor to PIL image"""
    img_tensor = tensor[0]
    img = img_tensor.permute(1, 2, 0).numpy()
    img = (img * 255).astype('uint8')
    return Image.fromarray(img)


# Streamlit app
# TODO: Here's a simple streamlit app. Try to see what you add
def main():    
    # Load models
    with st.spinner("Loading models..."):
        model, clip_model = load_models()
    
    if model is None or clip_model is None:
        st.stop()
    
    # st.form to only generate image when prompt is passed in
    with st.form(key="prompt_form"):
        text_prompt = st.text_input("Enter a description: ")
        submit_button = st.form_submit_button("Generate")
    
    # if input passed to st.form, generate image
    if submit_button and text_prompt:
        with st.spinner(f"Generating..."):
            samples = generate_faces(model, clip_model, text_prompt, num_samples=1)
            image = tensor_to_pil_image(samples) # Convert to PIL image
            st.image(image, use_container_width=True) # Display image

if __name__ == "__main__":
    main()

