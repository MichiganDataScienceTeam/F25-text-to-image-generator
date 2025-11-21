import streamlit as st
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image
import clip
from PIL import Image
import io
import os

st.set_page_config(
    page_title="Face Generation with CVAE",
    layout="centered"
)

# Device selection
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
    # TODO: change parameteres to the ones used in the training
    latent_size = 128 
    clip_dim = 512
    init_channels = 64 
    image_size = 64
    image_channels = 3
    
    # Load CVAE model
    model = CelebaCVAE(image_channels, init_channels, latent_size, clip_dim, image_size).to(device)
    
    # Load model weights
    model_path = 'new_celeba_cvae_model.pth'
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None, None
    
    st.info(f"Loading model from: {model_path}")
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load CLIP model
    clip_model, _ = clip.load("ViT-B/32", device=device)
    
    return model, clip_model

# Same as in cvae_celeba_solution.ipynb
def generate_faces(model, clip_model, text_prompt, num_samples=4, temperature=1.0):    
    text = clip.tokenize([text_prompt]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text)
    
    with torch.no_grad():
        # Use your latent_size value for second parameter of torch.randn
        sample_z = torch.randn(num_samples, 128).to(device) * temperature
        text_condition = text_features.repeat(num_samples, 1)
        samples = model.decode(sample_z, text_condition).cpu()
    
    return samples


def tensor_to_pil_images(tensor):
    """Convert tensor to list of PIL images"""
    images = []
    for i in range(tensor.size(0)):
        img_tensor = tensor[i]
        img = img_tensor.permute(1, 2, 0).numpy()
        img = (img * 255).astype('uint8')
        images.append(Image.fromarray(img))
    return images


# Main App
def main():
    st.title("Text to Image Generator")
    
    # chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Load models
    with st.spinner("Loading models..."):
        model, clip_model = load_models()
    
    if model is None or clip_model is None:
        st.stop()
        
    # settings
    with st.sidebar:
        st.header("Generation Settings")
        num_samples = st.slider("Number of samples", min_value=1, max_value=8, value=4)
        st.markdown("---")
        st.header("Chat History")
        if st.button("Clear History", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
        st.markdown(f"**Total generations:** {len(st.session_state.chat_history)}")
        st.markdown("---")

    
    # Text input
    text_prompt = st.text_input(
        "Enter a description for the face you want to generate:",
        value="smiling woman",
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        generate_btn = st.button("Generate Faces", type="primary", use_container_width=True)
    
    if generate_btn and text_prompt:
        with st.spinner(f"Generating faces for: '{text_prompt}'..."):
            samples = generate_faces(model, clip_model, text_prompt, num_samples)
            images = tensor_to_pil_images(samples)
            
            history_entry = {
                'prompt': text_prompt,
                'images': images,
                'num_samples': num_samples
            }
            st.session_state.chat_history.append(history_entry)
            
            st.markdown("---")
            st.subheader(f"Generated Faces: *{text_prompt}*")
            cols = st.columns(min(num_samples, 4))
            for idx, img in enumerate(images):
                with cols[idx % 4]:
                    st.image(img, use_container_width=True)
            
            st.markdown("---")
            
            import torchvision
            grid = torchvision.utils.make_grid(samples, nrow=min(num_samples, 4), normalize=True)
            grid_img = grid.permute(1, 2, 0).numpy()
            grid_img = (grid_img * 255).astype('uint8')
            grid_pil = Image.fromarray(grid_img)
    
    elif generate_btn and not text_prompt:
        st.warning("Please enter a text prompt!")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.header("History")
        
        for idx, entry in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"**Prompt:** {entry['prompt']}")
            cols = st.columns(min(entry['num_samples'], 4))
            for img_idx, img in enumerate(entry['images']):
                with cols[img_idx % 4]:
                    st.image(img, use_container_width=True)
            
            if idx < len(st.session_state.chat_history) - 1:
                st.markdown("---")


if __name__ == "__main__":
    main()

