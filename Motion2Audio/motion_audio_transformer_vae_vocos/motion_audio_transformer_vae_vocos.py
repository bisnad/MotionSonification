"""
that operates on the audio side with latent encodings of a pretrained VAE
the VAE has been trained on vocos generated MEL spectra
"""

"""
Imports
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from collections import OrderedDict
import scipy.linalg as sclinalg

import math
import os, sys, time, subprocess
import numpy as np
import csv
import matplotlib.pyplot as plt

# audio specific imports

import torchaudio
import torchaudio.transforms as transforms
import simpleaudio as sa

# vocos specific imports
from vocos import Vocos 

# mocap specific imports

from common import utils
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import mocap_tools as mocap
from common.quaternion import qmul, qrot, qnormalize_np, slerp
from common.pose_renderer import PoseRenderer

"""
Compute Device
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

"""
Mocap Settings
"""

mocap_data_path = "E:/data/mocap/Motion2Audio/stocos/fbx_50hz/"
mocap_data_files = ["Take_1_50fps_crop.fbx"]
mocap_valid_ranges = [[0, 16709]]

mocap_pos_scale = 0.1
mocap_fps = 50
mocap_dim = -1 # automatically determined when loading mocap data
mocap_input_seq_length = 64

load_mocap_stat = False
mocap_mean_file = "results/stat/mocap_mean.pt"
mocap_std_file = "results/stat/mocap_std.pt"


"""
Audio Settings
"""

audio_data_path = "E:/data/audio/Motion2Audio/stocos/"
audio_data_files = ["Take1__double_Bind_HQ_audio_crop_48khz.wav"]
audio_sample_rate = 48000
audio_channels = 1
audio_window_length = 2048 # this results in 9 mel spectra
audio_window_increment = audio_sample_rate // mocap_fps
audio_samples_per_mocap_frame = audio_window_increment
audio_latents_input_seq_length = mocap_input_seq_length
audio_waveform_input_seq_length = audio_latents_input_seq_length * audio_samples_per_mocap_frame + audio_window_length - audio_samples_per_mocap_frame


"""
Vocos Settings
"""

vocos_pretrained_config = "kittn/vocos-mel-48khz-alpha1"

"""
Vocos VAE Model Settings
"""

latent_dim = 32
audio_dim = latent_dim
vae_conv_channel_counts = [ 16, 32, 64, 128 ]
vae_conv_kernel_size = (5, 3)
vae_dense_layer_sizes = [ 512 ]

encoder_weights_file = "../../../audio/audio_autoencoder/audio_vae_vocos/results_Stocos_DoubleBind_2/weights/encoder_weights_epoch_400"
decoder_weights_file = "../../../audio/audio_autoencoder/audio_vae_vocos/results_Stocos_DoubleBind_2/weights/decoder_weights_epoch_400"

load_vocos_stat = False
vocos_latents_mean_file = "results/stat/vocos_latents_mean.pt"
vocos_latents_std_file = "results/stat/vocos_latents_std.pt"


"""
Dataset Settings
"""

mocap_frame_incr = 1
audio_sample_incr = int(1 / mocap_fps * audio_sample_rate) # same as audio_samples_per_mocap_frame
batch_size = 32
test_percentage = 0.1

"""
Model Settings
"""

transformer_layer_count = 6
transformer_head_count = 8
transformer_embed_dim = 256
transformer_dropout = 0.1   
pos_encoding_max_length = max(mocap_input_seq_length, audio_latents_input_seq_length)

load_weights = False
transformer_weights_file = "results_motionbank_soni2/weights/transformer_weights_epoch_400"

"""
Training Settings
"""



learning_rate = 1e-4
non_teacher_forcing_step_count = 10
model_save_interval = 50
load_weights = False
save_weights = True
transformer_load_weights_path = "results/weights/transformer_weights_epoch_200"
epochs = 400


"""
Mocap Visualisation Settings
"""

view_ele = 90.0
view_azi = -90.0
view_line_width = 1.0
view_size = 4.0

"""
Load Data - Mocap
"""

# load mocap data
bvh_tools = bvh.BVH_Tools()
fbx_tools = fbx.FBX_Tools()
mocap_tools = mocap.Mocap_Tools()

mocap_all_data = []

for mocap_data_file in mocap_data_files:

    if mocap_data_file.endswith(".bvh") or mocap_data_file.endswith(".BVH"):
        bvh_data = bvh_tools.load(mocap_data_path + "/" + mocap_data_file)
        mocap_data = mocap_tools.bvh_to_mocap(bvh_data)
    elif mocap_data_file.endswith(".fbx") or mocap_data_file.endswith(".FBX"):
        fbx_data = fbx_tools.load(mocap_data_path + "/" + mocap_data_file)
        mocap_data = mocap_tools.fbx_to_mocap(fbx_data)[0] # first skeleton only  
        
    mocap_data["skeleton"]["offsets"] *= mocap_pos_scale
    mocap_data["motion"]["pos_local"] *= mocap_pos_scale
    
    # set x and z offset of root joint to zero
    mocap_data["skeleton"]["offsets"][0, 0] = 0.0 
    mocap_data["skeleton"]["offsets"][0, 2] = 0.0 
    
    # set x and z offset of root joint to zero
    mocap_data["skeleton"]["offsets"][0, 0] = 0.0 
    mocap_data["skeleton"]["offsets"][0, 2] = 0.0
    
    if mocap_data_file.endswith(".bvh") or mocap_data_file.endswith(".BVH"):
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat_bvh(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])
    elif mocap_data_file.endswith(".fbx") or mocap_data_file.endswith(".FBX"):
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])

    mocap_all_data.append(mocap_data)
    
# get mocap info

mocap_skeleton = mocap_all_data[0]["skeleton"]

offsets = mocap_skeleton["offsets"].astype(np.float32)
parents = mocap_skeleton["parents"]
children = mocap_skeleton["children"]

mocap_motion = mocap_all_data[0]["motion"]["rot_local"]

joint_count = mocap_motion.shape[1]
joint_dim = mocap_motion.shape[2]
pose_dim = joint_count * joint_dim
mocap_dim = pose_dim

# create edge list
def get_edge_list(children):
    edge_list = []

    for parent_joint_index in range(len(children)):
        for child_joint_index in children[parent_joint_index]:
            edge_list.append([parent_joint_index, child_joint_index])
    
    return edge_list

edge_list = get_edge_list(children)

poseRenderer = PoseRenderer(edge_list)
    
# calc mean and std on all mocap data

if load_mocap_stat == True:
    mocap_mean = torch.load(mocap_mean_file)
    mocap_std = torch.load(mocap_std_file)
else:

    mocap_sequences_concat = [ mocap_data["motion"]["rot_local"] for mocap_data in mocap_all_data ]
    mocap_sequences_concat = np.concatenate(mocap_sequences_concat, axis=0)
    mocap_sequences_concat = mocap_sequences_concat.reshape(mocap_sequences_concat.shape[0], -1)
    
    mocap_mean = np.mean(mocap_sequences_concat, axis=0, keepdims=True)
    mocap_std = np.std(mocap_sequences_concat, axis=0, keepdims=True)
    
    print("mocap_mean s ", mocap_mean.shape)
    print("mocap_std s ", mocap_std.shape)
    
    torch.save(mocap_mean, mocap_mean_file)
    torch.save(mocap_std, mocap_std_file)


"""
Load Data - Audio
"""

audio_all_data = []

for audio_data_file in audio_data_files:    
    audio_data, _ = torchaudio.load(audio_data_path + audio_data_file)
        
    audio_all_data.append(audio_data)

"""
Load Vocos Model
"""

vocos = Vocos.from_pretrained("kittn/vocos-mel-48khz-alpha1").to(device)
vocos.eval()

vocoder_features = vocos.feature_extractor(torch.rand(size=(1, audio_window_length), dtype=torch.float32).to(device))
mel_count = vocoder_features.shape[-1]
mel_filter_count = vocoder_features.shape[1]

"""
Create Vocos VAE Model
"""

# create encoder model

class Encoder(nn.Module):
    
    def __init__(self, latent_dim, mel_count, mel_filter_count, conv_channel_counts, conv_kernel_size, dense_layer_sizes):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.mel_count = mel_count
        self.mel_filter_count = mel_filter_count
        self.conv_channel_counts = conv_channel_counts
        self.conv_kernel_size = conv_kernel_size
        self.dense_layer_sizes = dense_layer_sizes
        
        # create convolutional layers
        self.conv_layers = nn.ModuleList()
        
        stride = ((self.conv_kernel_size[0] - 1) // 2, (self.conv_kernel_size[1] - 1) // 2)
        
        #print("conv_kernel_size ", conv_kernel_size)
        #print("stride ", stride)
        
        padding = stride
        
        self.conv_layers.append(nn.Conv2d(1, conv_channel_counts[0], self.conv_kernel_size, stride=stride, padding=padding))
        self.conv_layers.append(nn.LeakyReLU(0.2))
        self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[0]))
        
        conv_layer_count = len(conv_channel_counts)
        for layer_index in range(1, conv_layer_count):
            self.conv_layers.append(nn.Conv2d(conv_channel_counts[layer_index-1], conv_channel_counts[layer_index], self.conv_kernel_size, stride=stride, padding=padding))
            self.conv_layers.append(nn.LeakyReLU(0.2))
            self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[layer_index]))

        self.flatten = nn.Flatten()
        
        # create dense layers
        self.dense_layers = nn.ModuleList()
        
        last_conv_layer_size_x = int(mel_filter_count // np.power(stride[0], len(conv_channel_counts)))
        last_conv_layer_size_y = int(mel_count // np.power(stride[1], len(conv_channel_counts)))
        
        #print("last_conv_layer_size_x ", last_conv_layer_size_x)
        #print("last_conv_layer_size_y ", last_conv_layer_size_y)
        
        preflattened_size = [conv_channel_counts[-1], last_conv_layer_size_x, last_conv_layer_size_y]
        
        #print("preflattened_size ", preflattened_size)
        
        dense_layer_input_size = conv_channel_counts[-1] * last_conv_layer_size_x * last_conv_layer_size_y
        
        #print("dense_layer_input_size ", dense_layer_input_size)
        #print("self.dense_layer_sizes[0] ", self.dense_layer_sizes[0])
        
        self.dense_layers.append(nn.Linear(dense_layer_input_size, self.dense_layer_sizes[0]))
        self.dense_layers.append(nn.ReLU())
        
        dense_layer_count = len(dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            self.dense_layers.append(nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index]))
            self.dense_layers.append(nn.ReLU())
            
        # create final dense layers
        self.fc_mu = nn.Linear(self.dense_layer_sizes[-1], self.latent_dim)
        self.fc_std = nn.Linear(self.dense_layer_sizes[-1], self.latent_dim)


    def forward(self, x):
        
        #print("x0 s ", x.shape)
        
        for lI, layer in enumerate(self.conv_layers):
            
            #print("conv layer ", lI, " x in ", x.shape)
            
            x = layer(x)
            
            #print("conv layer ", lI, " x out ", x.shape)
    
        #print("x1 s ", x.shape)
        
        x = self.flatten(x)
        
        #print("x2 s ", x.shape)

        for lI, layer in enumerate(self.dense_layers):
            
            #print("dense layer ", lI, " x in ", x.shape)
            
            x = layer(x)
            
            #print("dense layer ", lI, " x out ", x.shape)
            
        #print("x3 s ", x.shape)
        
        mu = self.fc_mu(x)
        std = self.fc_std(x)
        
        #print("mu s ", mu.shape, " lvar s ", std.shape)

        return mu, std
    
    def reparameterize(self, mu, std):
        z = mu + std*torch.randn_like(std)
        return z

encoder = Encoder(latent_dim, mel_count, mel_filter_count, vae_conv_channel_counts, vae_conv_kernel_size, vae_dense_layer_sizes).to(device)

print(encoder)

# load encoder weights

encoder.load_state_dict(torch.load(encoder_weights_file, map_location=device))

encoder.eval()

# Decoder 
class Decoder(nn.Module):
    
    def __init__(self, latent_dim, mel_count, mel_filter_count, conv_channel_counts, conv_kernel_size, dense_layer_sizes):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.mel_count = mel_count
        self.mel_filter_count = mel_filter_count
        self.conv_channel_counts = conv_channel_counts
        self.conv_kernel_size = conv_kernel_size
        self.dense_layer_sizes = dense_layer_sizes
        
        # create dense layers
        self.dense_layers = nn.ModuleList()
        
        stride = ((self.conv_kernel_size[0] - 1) // 2, (self.conv_kernel_size[1] - 1) // 2)
        
        print("stride ", stride)
                
        self.dense_layers.append(nn.Linear(latent_dim, self.dense_layer_sizes[0]))
        self.dense_layers.append(nn.ReLU())
        
        dense_layer_count = len(dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            self.dense_layers.append(nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index]))
            self.dense_layers.append(nn.ReLU())
            
        last_conv_layer_size_x = int(mel_filter_count // np.power(stride[0], len(conv_channel_counts)))
        last_conv_layer_size_y = int(mel_count // np.power(stride[1], len(conv_channel_counts)))
        
        #print("last_conv_layer_size_x ", last_conv_layer_size_x)
        #print("last_conv_layer_size_y ", last_conv_layer_size_y)
        
        preflattened_size = [conv_channel_counts[0], last_conv_layer_size_x, last_conv_layer_size_y]
        
        #print("preflattened_size ", preflattened_size)
        
        dense_layer_output_size = conv_channel_counts[0] * last_conv_layer_size_x * last_conv_layer_size_y
        
        #print("dense_layer_output_size ", dense_layer_output_size)

        self.dense_layers.append(nn.Linear(self.dense_layer_sizes[-1], dense_layer_output_size))
        self.dense_layers.append(nn.ReLU())

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=preflattened_size)
        
        # create convolutional layers
        self.conv_layers = nn.ModuleList()
        
        padding = stride
        output_padding = (padding[0] - 1, padding[1] - 1) # does this universally work?
        
        conv_layer_count = len(conv_channel_counts)
        for layer_index in range(1, conv_layer_count):
            self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[layer_index-1]))
            self.conv_layers.append(nn.ConvTranspose2d(conv_channel_counts[layer_index-1], conv_channel_counts[layer_index], self.conv_kernel_size, stride=stride, padding=padding, output_padding=output_padding))
            self.conv_layers.append(nn.LeakyReLU(0.2))
            
        self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[-1]))
        self.conv_layers.append(nn.ConvTranspose2d(conv_channel_counts[-1], 1, self.conv_kernel_size, stride=stride, padding=padding, output_padding=output_padding))

    def forward(self, x):
        
        #print("x0 s ", x.shape)
        
        for lI, layer in enumerate(self.dense_layers):
            
            #print("dense layer ", lI, " x in ", x.shape)
            
            x = layer(x)
            
            #print("dense layer ", lI, " x out ", x.shape)
            
        #print("x1 s ", x.shape)
        
        x = self.unflatten(x)
        
        #print("x2 s ", x.shape)

        for lI, layer in enumerate(self.conv_layers):
            
            #print("conv layer ", lI, " x in ", x.shape)
            
            x = layer(x)
            
            #print("conv layer ", lI, " x out ", x.shape)
    
        #print("x3 s ", x.shape)

        return x
    
vae_conv_channel_counts_reversed = vae_conv_channel_counts.copy()
vae_conv_channel_counts_reversed.reverse()
    
vae_dense_layer_sizes_reversed = vae_dense_layer_sizes.copy()
vae_dense_layer_sizes_reversed.reverse()

vae_conv_channel_counts_reversed

decoder = Decoder(latent_dim, mel_count, mel_filter_count, vae_conv_channel_counts_reversed, vae_conv_kernel_size, vae_dense_layer_sizes_reversed).to(device)

print(decoder)

# load decoder weights

decoder.load_state_dict(torch.load(decoder_weights_file, map_location=device))

decoder.eval()

# compute and normalise vocos vae latents

if load_vocos_stat == True:
    vocos_latents_mean = torch.load(vocos_latents_mean_file)
    vocos_latents_std = torch.load(vocos_latents_std_file)
else:
    vocos_latents_all_data = []
    
    for audio_data in audio_all_data:    
        
        #print("audio_data s ", audio_data.shape)
        
        audio_sample_count = audio_data.shape[1]
        
        for asI in range(0, audio_sample_count - audio_window_length, audio_window_increment):
            
            audio_excerpt = audio_data[:, asI:asI+audio_window_length].to(device)
            
            #print("audio_excerpt s ", audio_excerpt.shape)
            
            with torch.no_grad():
                audio_mels = vocos.feature_extractor(audio_excerpt.unsqueeze(0))
                
            #print("audio_mels s ", audio_mels.shape)
            
            #audio_mels = audio_mels.squeeze(0)
            
            #print("audio_mels s ", audio_mels.shape)
            
            with torch.no_grad():
                encoder_out_mu, encoder_out_std = encoder(audio_mels)
                encoder_out_std = torch.nn.functional.softplus(encoder_out_std) + 1e-8
                audio_encoder_out = encoder.reparameterize(encoder_out_mu, encoder_out_std)
                
                #print("audio_encoder_out s ", audio_encoder_out.shape)
    
                vocos_latents_all_data.append(audio_encoder_out)
      
    vocos_latents_all_data = torch.cat(vocos_latents_all_data, dim=0)
    
    vocos_latents_mean = torch.mean(vocos_latents_all_data, dim=0, keepdim=True)
    vocos_latents_std = torch.std(vocos_latents_all_data, dim=0, keepdim=True)
    
    print("vocos_latents_mean s ", vocos_latents_mean.shape)
    print("vocos_latents_std s ", vocos_latents_std.shape)
    
    torch.save(vocos_latents_mean, vocos_latents_mean_file)
    torch.save(vocos_latents_std, vocos_latents_std_file)

"""
Create Dataset
"""

X_mocap = []
Y_audio = []

for sI in range(len(mocap_all_data)):
    
    mocap_data = mocap_all_data[sI]["motion"]["rot_local"].reshape(-1, pose_dim)
    audio_data = audio_all_data[sI][0]

    #print(sI)
    print("mocap_data s ", mocap_data.shape)
    print("audio_data s ", audio_data.shape)
    
    mocap_frame_count = mocap_data.shape[0]
    
    for mfI in range(3, mocap_frame_count - mocap_input_seq_length - 2 - non_teacher_forcing_step_count, mocap_frame_incr):
        
        # mocap sequence part
        
        # get mocap sequence
        mocap_excerpt_start = mfI
        mocap_excerpt_end = mfI + mocap_input_seq_length + non_teacher_forcing_step_count
        
        #print("mocap_excerpt_start ", mocap_excerpt_start)
        #print("mocap_excerpt_end ", mocap_excerpt_end)
        
        mocap_excerpt = mocap_data[mocap_excerpt_start:mocap_excerpt_end, :]
        
        #print("mfI ", mfI, " me s ", mocap_excerpt.shape)
        
        # normalise mocap sequence
        mocap_excerpt_norm = (mocap_excerpt - mocap_mean) / (mocap_std + 1e-8) 
        
        X_mocap.append(np.expand_dims(mocap_excerpt_norm, 0))
        
        # audio sequence part
        
        audio_latents = []
        
        for alI in range(-1, audio_latents_input_seq_length + non_teacher_forcing_step_count, 1):
            
            #print("mfI ", mfI, " alI ", alI)
            
            audio_window_start = (mfI + alI) * audio_samples_per_mocap_frame - (audio_window_length - audio_samples_per_mocap_frame) // 2
            audio_window_end = audio_window_start + audio_window_length
            
            #print("aws ", audio_window_start, " awe ", audio_window_end)
            
            audio_window = audio_data[audio_window_start:audio_window_end].to(device)
            
            #print("audio_window s ", audio_window.shape)
            
            with torch.no_grad():
                audio_mels = vocos.feature_extractor(audio_window.reshape(1, 1, -1))
                encoder_out_mu, encoder_out_std = encoder(audio_mels)
                encoder_out_std = torch.nn.functional.softplus(encoder_out_std) + 1e-8
                audio_encoder_out = encoder.reparameterize(encoder_out_mu, encoder_out_std)
                
                #print("audio_encoder_out s ", audio_encoder_out.shape)
                
            audio_latents.append(audio_encoder_out)
        
        audio_latents = torch.cat(audio_latents, dim=0)
            
        #print("mfI ", mfI, " me s ", mocap_excerpt.shape, " audio_latents s ", audio_latents.shape)
        #print("audio_latents s ", audio_latents.shape)

        audio_latents_norm = (audio_latents - vocos_latents_mean) / (vocos_latents_std  + 1e-8) 

        Y_audio.append(audio_latents_norm.unsqueeze(0))  
    

X_mocap = np.concatenate(X_mocap, axis=0)
Y_audio = torch.cat(Y_audio, axis=0)

X_mocap = torch.from_numpy(X_mocap).to(torch.float32)
Y_audio = Y_audio.to(torch.float32)

#Y_audio = Y_audio.permute(0, 2, 1)

print("X_mocap s ", X_mocap.shape)
print("Y_audio s ", Y_audio.shape)

class SequenceDataset(Dataset):
    def __init__(self, X_mocap, Y_audio):
        self.X_mocap = X_mocap
        self.Y_audio = Y_audio
    
    def __len__(self):
        return self.X_mocap.shape[0]
    
    def __getitem__(self, idx):
        return self.X_mocap[idx, ...], self.Y_audio[idx, ...]

full_dataset = SequenceDataset(X_mocap, Y_audio)

x_item_mocap, y_item_audio = full_dataset[0]

print("x_item_mocap s ", x_item_mocap.shape)
print("y_item_audio s ", y_item_audio.shape)

test_size = int(test_percentage * len(full_dataset))
train_size = len(full_dataset) - test_size

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

x_batch_mocap, y_batch_audio = next(iter(train_loader))

print("x_batch_mocap s ", x_batch_mocap.shape)
print("y_batch_audio s ", y_batch_audio.shape)

"""
Create Models - PositionalEncoding
"""

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


"""
Create Models - Transformer
"""


class Transformer(nn.Module):

    # Constructor
    def __init__(
        self,
        mocap_dim,
        audio_dim,
        embed_dim,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
        pos_encoding_max_length
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # LAYERS
        self.mocap2embed = nn.Linear(mocap_dim, embed_dim) # map mocap data to embedding
        self.audio2embed = nn.Linear(audio_dim, embed_dim) # map audio data to embedding

        self.positional_encoder = PositionalEncoding(
            dim_model=embed_dim, dropout_p=dropout_p, max_len=pos_encoding_max_length
        )
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = num_encoder_layers)
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers = num_decoder_layers)
        
        self.embed2audio = nn.Linear(embed_dim, audio_dim) # map embedding to audio data
        
    def get_src_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.ones(size, size)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        return mask
       
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask
        
       
    def forward(self, mocap_data, audio_data):
        
        #print("forward")
        
        #print("data s ", data.shape)

        src_mask = self.get_src_mask(mocap_data.shape[1]).to(mocap_data.device)
        tgt_mask = self.get_tgt_mask(audio_data.shape[1]).to(audio_data.device)
        
        mocap_embedded = self.mocap2embed(mocap_data) * math.sqrt(self.embed_dim)
        mocap_embedded = self.positional_encoder(mocap_embedded)
        
        audio_embedded = self.audio2embed(audio_data) * math.sqrt(self.embed_dim)
        audio_embedded = self.positional_encoder(audio_embedded)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        encoder_out = self.encoder(mocap_embedded, mask=src_mask)
        decoder_out = self.decoder(audio_embedded, encoder_out, tgt_mask =tgt_mask)
        
        out = self.embed2audio(decoder_out)
        
        return out

transformer = Transformer(mocap_dim=mocap_dim, 
                          audio_dim=audio_dim,
                          embed_dim=transformer_embed_dim, 
                          num_heads=transformer_head_count, 
                          num_encoder_layers=transformer_layer_count, 
                          num_decoder_layers=transformer_layer_count, 
                          dropout_p=transformer_dropout,
                          pos_encoding_max_length=pos_encoding_max_length).to(device)

print(transformer)


if load_weights and transformer_load_weights_path:
    transformer.load_state_dict(torch.load(transformer_load_weights_path, map_location=device))


# test model

x_batch_mocap = x_batch_mocap[:, :mocap_input_seq_length, :].to(device)
y_batch_audio = y_batch_audio[:, :audio_latents_input_seq_length, :].to(device)

yhat_batch_audio = transformer(x_batch_mocap, y_batch_audio)

print("x_batch_mocap s ", x_batch_mocap.shape)
print("y_batch_audio s ", y_batch_audio.shape)
print("yhat_batch_audio s ", yhat_batch_audio.shape)

"""
Training
"""

optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1) # reduce the learning every 20 epochs by a factor of 10

n1_loss = nn.L1Loss()
mse_loss = nn.MSELoss()

def audio_rec_loss(y, yhat):
    flat_y = torch.flatten(y)
    flat_yhat = torch.flatten(yhat)

    #_loss = n1_loss(flat_yhat, flat_y)
    _loss = mse_loss(flat_yhat, flat_y)
    
    return _loss

def loss(y, yhat):
    
    _loss = audio_rec_loss(y, yhat)
    
    return _loss

def train_step(x_mocap, y_audio):

    #print("x_mocap s ", x_mocap.shape)
    #print("y_audio s ", y_audio.shape)

    # teacher forcing step
    #print("teacher forcing step")
    x_mocap_input = x_mocap[:, :mocap_input_seq_length, :]
    y_audio_input = y_audio[:, :audio_latents_input_seq_length, :]
    y_audio_target = y_audio[:, 1:audio_latents_input_seq_length+1, :] # shift target by one so we predict next token
    
    #print("x_mocap_input s ", x_mocap_input.shape)
    #print("y_audio_input s ", y_audio_input.shape)
    #print("y_audio_target s ", y_audio_target.shape)
    
    yhat_audio_target = transformer(x_mocap_input, y_audio_input)
    
    #print("yhat_audio_target s ", yhat_audio_target.shape)
    
    _loss = loss(y_audio_target, yhat_audio_target) 
    
    # non-teacher_forcing steps
    
    y_audio_input = yhat_audio_target.detach().clone()
    
    for i in range(non_teacher_forcing_step_count):
        #print("non teacher forcing step ", i)
        
        x_mocap_input = x_mocap[:, i:mocap_input_seq_length+i, :]
        y_audio_target = y_audio[:, i+1:audio_latents_input_seq_length+i+1, :] # shift target by one so we predict next token
        yhat_audio_target = transformer(x_mocap_input, y_audio_input)
        _loss += loss(y_audio_target, yhat_audio_target) 
        
        y_audio_input = yhat_audio_target.detach().clone()

    _loss /= (non_teacher_forcing_step_count + 1)

    # Backpropagation
    optimizer.zero_grad()
    _loss.backward()
    optimizer.step()

    return _loss


"""
x_batch_mocap, y_batch_audio = next(iter(train_loader))
x_batch_mocap = x_batch_mocap.to(device)
y_batch_audio = y_batch_audio.to(device)
_loss = train_step(x_batch_mocap, y_batch_audio)
"""

def test_step(x_mocap, y_audio):
    
    transformer.eval()

    #print("x_mocap s ", x_mocap.shape)
    #print("y_audio s ", y_audio.shape)

    # teacher forcing step
    #print("teacher forcing step")
    x_mocap_input = x_mocap[:, :mocap_input_seq_length, :]
    y_audio_input = y_audio[:, :audio_latents_input_seq_length, :]
    y_audio_target = y_audio[:, 1:audio_latents_input_seq_length+1, :] # shift target by one so we predict next token
    
    #print("x_mocap_input s ", x_mocap_input.shape)
    #print("y_audio_input s ", y_audio_input.shape)
    #print("y_audio_target s ", y_audio_target.shape)
    
    with torch.no_grad():
        yhat_audio_target = transformer(x_mocap_input, y_audio_input)
        #print("yhat_audio_target s ", yhat_audio_target.shape)
        _loss = loss(y_audio_target, yhat_audio_target) 
    
    # non-teacher_forcing steps
    
    y_audio_input = yhat_audio_target.detach().clone()
    
    for i in range(non_teacher_forcing_step_count):
        #print("non teacher forcing step ", i)
        
        x_mocap_input = x_mocap[:, i:mocap_input_seq_length+i, :]
        y_audio_target = y_audio[:, i+1:audio_latents_input_seq_length+i+1, :] # shift target by one so we predict next token
        
        with torch.no_grad():
            yhat_audio_target = transformer(x_mocap_input, y_audio_input)
            _loss += loss(y_audio_target, yhat_audio_target) 
        
        y_audio_input = yhat_audio_target.detach().clone()

    _loss /= (non_teacher_forcing_step_count + 1)

    
    transformer.train()

    return _loss

def train(train_dataloader, test_dataloader, epochs):
    
    loss_history = {}
    loss_history["train"] = []
    loss_history["test"] = []

    for epoch in range(epochs):
        start = time.time()
        
        _train_loss_per_epoch = []

        for train_batch in train_dataloader:
            x_mocap = train_batch[0].to(device)
            y_audio = train_batch[1].to(device)
            
            _loss = train_step(x_mocap, y_audio)
            
            _loss = _loss.detach().cpu().numpy()
            
            _train_loss_per_epoch.append(_loss)

        _train_loss_per_epoch = np.mean(np.array(_train_loss_per_epoch))

        _test_loss_per_epoch = []
        
        for test_batch in test_dataloader:
            x_mocap = test_batch[0].to(device)
            y_audio = test_batch[1].to(device)
            
            _loss = test_step(x_mocap, y_audio)

            _loss = _loss.detach().cpu().numpy()
            
            _test_loss_per_epoch.append(_loss)
        
        _test_loss_per_epoch = np.mean(np.array(_test_loss_per_epoch))
        
        if epoch % model_save_interval == 0 and save_weights == True:
            torch.save(transformer.state_dict(), "results/weights/transformer_weights_epoch_{}".format(epoch))
        
        loss_history["train"].append(_train_loss_per_epoch)
        loss_history["test"].append(_test_loss_per_epoch)
        
        scheduler.step()
        
        print ('epoch {} : train: {:01.4f} test: {:01.4f} time {:01.2f}'.format(epoch + 1, _train_loss_per_epoch, _test_loss_per_epoch, time.time()-start))
    
    return loss_history

# fit model
loss_history = train(train_loader, test_loader, epochs)

# save history
def save_loss_as_image(loss_history, image_file_name):
    keys = list(loss_history.keys())
    epochs = len(loss_history[keys[0]])
    
    for key in keys:
        plt.plot(range(epochs), loss_history[key], label=key)
        
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig(image_file_name)

def save_loss_as_csv(loss_history, csv_file_name):
    with open(csv_file_name, 'w') as csv_file:
        csv_columns = list(loss_history.keys())
        csv_row_count = len(loss_history[csv_columns[0]])
        
        
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns, delimiter=',', lineterminator='\n')
        csv_writer.writeheader()
    
        for row in range(csv_row_count):
        
            csv_row = {}
        
            for key in loss_history.keys():
                csv_row[key] = loss_history[key][row]

            csv_writer.writerow(csv_row)


save_loss_as_csv(loss_history, "results/histories/history_{}.csv".format(epochs))
save_loss_as_image(loss_history, "results/histories/history_{}.png".format(epochs))

# save model weights
torch.save(transformer.state_dict(), "results/weights/transformer_weights_epoch_{}".format(epochs))

# inference

# TODO

audio_window_env = torch.hann_window(audio_window_length)

def create_ref_audio(waveform_data, mocap_start_frame_index, mocap_frame_count, file_name):
    
    waveform_start_index = mocap_start_frame_index * audio_sample_incr
    waveform_end_index = waveform_start_index + mocap_frame_count * audio_sample_incr
    
    print("waveform_start_index ", waveform_start_index, " waveform_end_index ", waveform_end_index)
    
    ref_audio = waveform_data[waveform_start_index:waveform_end_index]
    
    torchaudio.save(file_name, ref_audio.unsqueeze(0), audio_sample_rate)
   
    
create_ref_audio(audio_all_data[0][0], 60 * mocap_fps, 10 * mocap_fps, "ref_audio.wav")

def create_ref_audio2(waveform_data, mocap_start_frame_index, mocap_frame_count, file_name):
    
    #print("waveform_data s ", waveform_data.shape)
    
    waveform_start_index = mocap_start_frame_index * audio_samples_per_mocap_frame
    waveform_end_index = waveform_start_index + mocap_frame_count * audio_samples_per_mocap_frame
    waveform_start_index = waveform_start_index - (audio_window_length - audio_samples_per_mocap_frame) // 2
    waveform_end_index = waveform_end_index + (audio_window_length - audio_samples_per_mocap_frame) // 2
    
    audio_sample_count = waveform_end_index - waveform_start_index
    
    rec_audio = torch.zeros(audio_sample_count, dtype=torch.float32)

    #print("waveform_start_index ", waveform_start_index, " waveform_end_index ", waveform_end_index)
    #print("rec_audio s ", rec_audio.shape)
    
    for asI in range(0, audio_sample_count - audio_window_length, audio_samples_per_mocap_frame):
        
        #print("asI ", asI)
        
        audio_window = waveform_data[waveform_start_index + asI:waveform_start_index + asI + audio_window_length].to(device)
        
        #print("audio_window s ", audio_window.shape)
        
        with torch.no_grad():
            audio_mels = vocos.feature_extractor(audio_window.reshape(1, 1, -1))
            
            #print("audio_mels s ", audio_mels.shape)

            encoder_out_mu, encoder_out_std = encoder(audio_mels)
            encoder_out_std = torch.nn.functional.softplus(encoder_out_std) + 1e-8
            audio_encoder_out = encoder.reparameterize(encoder_out_mu, encoder_out_std)
            
            #print("audio_encoder_out s ", audio_encoder_out.shape)
            
            audio_decoder_out = decoder(audio_encoder_out)
            
            #print("audio_decoder_out s ", audio_decoder_out.shape)
            
            voc_audio = vocos.decode(audio_decoder_out.squeeze(1))
            
            #print("voc_audio s ", voc_audio.shape)
            
        voc_audio = voc_audio.detach().cpu().reshape(-1)
        
        rec_audio[asI:asI + audio_window_length] += voc_audio * audio_window_env
    
    torchaudio.save(file_name, rec_audio.unsqueeze(0), audio_sample_rate)

create_ref_audio2(audio_all_data[0][0], 60 * mocap_fps, 10 * mocap_fps, "rec_audio.wav")
    
def forward_kinematics(rotations, root_positions):
    """
    Perform forward kinematics using the given trajectory and local rotations.
    Arguments (where N = batch size, L = sequence length, J = number of joints):
     -- rotations: (N, L, J, 4) tensor of unit quaternions describing the local rotations of each joint.
     -- root_positions: (N, L, 3) tensor describing the root joint positions.
    """

    assert len(rotations.shape) == 4
    assert rotations.shape[-1] == 4
    
    toffsets = torch.tensor(offsets).to(device)
    
    positions_world = []
    rotations_world = []

    expanded_offsets = toffsets.expand(rotations.shape[0], rotations.shape[1], offsets.shape[0], offsets.shape[1])

    # Parallelize along the batch and time dimensions
    for jI in range(offsets.shape[0]):
        if parents[jI] == -1:
            positions_world.append(root_positions)
            rotations_world.append(rotations[:, :, 0])
        else:
            positions_world.append(qrot(rotations_world[parents[jI]], expanded_offsets[:, :, jI]) \
                                   + positions_world[parents[jI]])
            if len(children[jI]) > 0:
                rotations_world.append(qmul(rotations_world[parents[jI]], rotations[:, :, jI]))
            else:
                # This joint is a terminal node -> it would be useless to compute the transformation
                rotations_world.append(None)

    return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)


def create_ref_anim(mocap_data, mocap_start_frame_index, mocap_frame_count, file_name):
    
    pose_sequence = mocap_data[mocap_start_frame_index:mocap_start_frame_index + mocap_frame_count]

    pose_count = pose_sequence.shape[0]
    pose_sequence = np.reshape(pose_sequence, (pose_count, joint_count, joint_dim))
    
    pose_sequence = torch.tensor(np.expand_dims(pose_sequence, axis=0)).to(device)
    zero_trajectory = torch.tensor(np.zeros((1, pose_count, 3), dtype=np.float32)).to(device)
    
    skel_sequence = forward_kinematics(pose_sequence, zero_trajectory)
    
    skel_sequence = skel_sequence.detach().cpu().numpy()
    skel_sequence = np.squeeze(skel_sequence)    
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence)
    skel_images = poseRenderer.create_pose_images(skel_sequence, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    skel_images[0].save(file_name, save_all=True, append_images=skel_images[1:], optimize=False, duration=1000 / mocap_fps, loop=0)
    
    

test_mocap_data = torch.from_numpy(mocap_all_data[0]["motion"]["rot_local"]).to(torch.float32)
test_mocap_data = test_mocap_data.reshape(-1, pose_dim)

test_mocap_data.shape

create_ref_anim(test_mocap_data, 60 * mocap_fps, 10 * mocap_fps, "ref3_mocap.gif")

def create_pred_audio(mocap_data, audio_data, mocap_start_frame_index, mocap_frame_count, file_name):
    
    transformer.eval()
    
    # get mocap excerpt
    mocap_excerpt = mocap_data[mocap_start_frame_index:mocap_start_frame_index + mocap_input_seq_length, :]
    #print("mocap_excerpt s ", mocap_excerpt.shape)
    
    # normalise mocap data
    mocap_excerpt_norm = (mocap_excerpt - mocap_mean) / (mocap_std + 1e-8) 
    
    # mocap input sequence
    x_mocap =  torch.tensor(mocap_excerpt_norm, dtype=torch.float32).unsqueeze(0).to(device)
    #print("x_mocap s ", x_mocap.shape)
    
    # get initial audio excerpt
    waveform_start_index = mocap_start_frame_index * audio_samples_per_mocap_frame
    waveform_start_index = waveform_start_index - (audio_window_length - audio_samples_per_mocap_frame) // 2
    waveform_end_index = waveform_start_index + audio_window_length
    
    #print("waveform_start_index ", waveform_start_index, " waveform_end_index ", waveform_end_index)
    
    audio_waveform_excerpt = audio_data[waveform_start_index:waveform_end_index].to(device)
    #print("audio_waveform_excerpt s ", audio_waveform_excerpt.shape)
    
    # convert initial audio excerpt to vocos latens
    with torch.no_grad():
        audio_mels_excerpt = vocos.feature_extractor(audio_waveform_excerpt.reshape(1, 1, -1))
        #print("audio_mels_excerpt s ", audio_mels_excerpt.shape)
        encoder_out_mu, encoder_out_std = encoder(audio_mels_excerpt)
        encoder_out_std = torch.nn.functional.softplus(encoder_out_std) + 1e-8
        #print("encoder_out_mu s ", encoder_out_mu.shape)
        audio_latents_excerpt = encoder.reparameterize(encoder_out_mu, encoder_out_std)
        #print("audio_latents_excerpt s ", audio_latents_excerpt.shape)
        
    # normalise vocos latens
    audio_latents_excerpt_norm = (audio_latents_excerpt - vocos_latents_mean) / (vocos_latents_std + 1e-8) 
    
    # audio input sequence
    y_audio = audio_latents_excerpt_norm.unsqueeze(0)
    #print("y_audio s ", y_audio.shape)
    
    for i in range(audio_latents_input_seq_length):
        #print("i ", i)
        
        with torch.no_grad():
            
            #print("y_audio 1 s ", y_audio.shape)
            
            yhat_audio = transformer(x_mocap, y_audio).detach()
            
            #print("yhat_audio s ", yhat_audio.shape)
            
            # Concatenate previous input with last predicted output
            y_audio = torch.cat((y_audio, yhat_audio[:, -1:, :]), dim=1)
            
            #print("y_audio 2 s ", y_audio.shape)

    gen_vocos_latents_norm = y_audio.detach().squeeze(0)
    #print("gen_vocos_latents_norm s ", gen_vocos_latents_norm.shape)
    
    # denormalise vocos latents
    gen_vocos_latents = gen_vocos_latents_norm * vocos_latents_std + vocos_latents_mean
    
    # generate audio waveforms
    with torch.no_grad():
            gen_audio_mels = decoder(gen_vocos_latents)
            #print("gen_audio_mels 1 s ", gen_audio_mels.shape)
            gen_audio_mels = gen_audio_mels.squeeze(1)
            #print("gen_audio_mels 2 s ", gen_audio_mels.shape)
            gen_audio_waveforms = vocos.decode(gen_audio_mels)
            #print("gen_audio_waveforms s ", gen_audio_waveforms.shape)
    
    # generate final audio waveform
    gen_audio_waveforms = gen_audio_waveforms.detach().cpu()
    gen_audio = torch.zeros(gen_audio_waveforms.shape[0] * audio_samples_per_mocap_frame + audio_window_length)

    asI = 0
    for gen_audio_waveform in gen_audio_waveforms:
        
        #print("gen_audio_waveform s ", gen_audio_waveform.shape)
        
        gen_audio[asI:asI + audio_window_length] += gen_audio_waveform * audio_window_env
        
        asI += audio_window_increment
    
    #print("gen_audio s ", gen_audio.shape)
    
    torchaudio.save(file_name, gen_audio.unsqueeze(0), audio_sample_rate)
    
    transformer.train()

test_mocap_data = torch.from_numpy(mocap_all_data[0]["motion"]["rot_local"]).to(torch.float32)
test_mocap_data = test_mocap_data.reshape(-1, pose_dim)
test_audio_data = audio_all_data[0][0]

print("test_mocap_data s ", test_mocap_data.shape)
print("test_audio_data s ", test_audio_data.shape)
    
create_pred_audio(test_mocap_data, test_audio_data, 60 * mocap_fps, 10 * mocap_fps, "gen_audio.wav")


def create_pred_audio3(mocap_data, audio_data, mocap_start_frame_index, mocap_frame_count, file_name):
    
    transformer.eval()
    
    gen_audio = torch.zeros(mocap_frame_count * audio_sample_incr + audio_window_length, dtype=torch.float32)
   
    # get initial audio excerpt
    waveform_start_index = mocap_start_frame_index * audio_samples_per_mocap_frame
    waveform_start_index = waveform_start_index - (audio_window_length - audio_samples_per_mocap_frame) // 2
    waveform_end_index = waveform_start_index + audio_window_length
    #print("waveform_start_index ", waveform_start_index, " waveform_end_index ", waveform_end_index)
    
    audio_waveform_excerpt = audio_data[waveform_start_index:waveform_end_index].to(device)
    #print("audio_waveform_excerpt s ", audio_waveform_excerpt.shape)
    
    # convert initial audio excerpt to vocos latens
    with torch.no_grad():
        audio_mels_excerpt = vocos.feature_extractor(audio_waveform_excerpt.reshape(1, 1, -1))
        #print("audio_mels_excerpt s ", audio_mels_excerpt.shape)
        encoder_out_mu, encoder_out_std = encoder(audio_mels_excerpt)
        encoder_out_std = torch.nn.functional.softplus(encoder_out_std) + 1e-8
        #print("encoder_out_mu s ", encoder_out_mu.shape)
        audio_latents_excerpt = encoder.reparameterize(encoder_out_mu, encoder_out_std)
        #print("audio_latents_excerpt s ", audio_latents_excerpt.shape)
        
    # normalise vocos latens
    audio_latents_excerpt_norm = (audio_latents_excerpt - vocos_latents_mean) / (vocos_latents_std + 1e-8) 
    
    # audio input sequence
    y_audio = audio_latents_excerpt_norm.unsqueeze(0)
    #print("y_audio s ", y_audio.shape)
    
    asI = 0
    for mfI in range(mocap_start_frame_index, mocap_start_frame_index + mocap_frame_count, 1):
        
        #print("mfI ", mfI, " asI ", asI)
        
        # get mocap excerpt
        mocap_start_frame_index = mfI
        mocap_excerpt = mocap_data[mocap_start_frame_index:mocap_start_frame_index + mocap_input_seq_length, :]
        #print("mocap_excerpt s ", mocap_excerpt.shape)
        
        # normalise mocap data
        mocap_excerpt_norm = (mocap_excerpt - mocap_mean) / (mocap_std + 1e-8) 
           
        # mocap input sequence
        x_mocap =  torch.tensor(mocap_excerpt_norm, dtype=torch.float32).unsqueeze(0).to(device)
        #print("x_mocap s ", x_mocap.shape)
        
        # predict next vocos latent
        with torch.no_grad():
            #print("y_audio s ", y_audio.shape)
            yhat_audio = transformer(x_mocap, y_audio).detach()
            #print("yhat_audio s ", yhat_audio.shape)
        
        y_audio = yhat_audio.clone()
            
        gen_vocos_latents_norm = y_audio.detach().squeeze(0)
        #print("gen_vocos_latents_norm s ", gen_vocos_latents_norm.shape)
            
        # denormalise vocos latents
        gen_vocos_latents = gen_vocos_latents_norm * vocos_latents_std + vocos_latents_mean
        
        # generate audio waveforms
        with torch.no_grad():
            gen_audio_mels = decoder(gen_vocos_latents)
            #print("gen_audio_mels 1 s ", gen_audio_mels.shape)
            gen_audio_mels = gen_audio_mels.squeeze(1)
            #print("gen_audio_mels 2 s ", gen_audio_mels.shape)
            gen_audio_waveforms = vocos.decode(gen_audio_mels)
            #print("gen_audio_waveforms s ", gen_audio_waveforms.shape)
            
        gen_audio_waveforms = gen_audio_waveforms[0].detach().cpu()
        
        gen_audio[asI:asI+audio_window_length] += gen_audio_waveforms * audio_window_env

        asI += audio_window_increment

    torchaudio.save(file_name, gen_audio.unsqueeze(0), audio_sample_rate)
       
    transformer.train()
        
test_mocap_data = torch.from_numpy(mocap_all_data[0]["motion"]["rot_local"]).to(torch.float32)
test_mocap_data = test_mocap_data.reshape(-1, pose_dim)
test_audio_data = audio_all_data[0][0]

print("test_mocap_data s ", test_mocap_data.shape)
print("test_audio_data s ", test_audio_data.shape)

test_mocap_start_times = [100, 200, 300, 400]
test_mocap_duration = 10
    
for test_mocap_start_time in test_mocap_start_times:
    create_ref_anim(test_mocap_data, test_mocap_start_time * mocap_fps, 10 * mocap_fps, "results/anims/_mocap_{}.gif".format(test_mocap_start_time))
    create_ref_audio(test_audio_data, test_mocap_start_time * mocap_fps, 10 * mocap_fps, "results/audio/ref_audio_{}-{}.wav".format(test_mocap_start_time, (test_mocap_start_time + test_mocap_duration)))
    create_ref_audio2(test_audio_data, test_mocap_start_time * mocap_fps, 10 * mocap_fps, "results/audio/ref_audio2_{}-{}.wav".format(test_mocap_start_time, (test_mocap_start_time + test_mocap_duration)))
    create_pred_audio3(test_mocap_data, test_audio_data, test_mocap_start_time * mocap_fps, 10 * mocap_fps, "results/audio/gen_audio_{}-{}_epoch_{}.wav".format(test_mocap_start_time, (test_mocap_start_time + test_mocap_duration), epochs))

"""
Alternative mocap data
"""

test_mocap_data_file = "E:/data/mocap/Motion2Audio/stocos/fbx_50hz/Take_2_50fps.fbx"
test_mocap_valid_ranges = [0, 11540]


if test_mocap_data_file.endswith(".bvh") or test_mocap_data_file.endswith(".BVH"):
    bvh_data = bvh_tools.load(test_mocap_data_file)
    test_mocap_data = mocap_tools.bvh_to_mocap(bvh_data)
elif test_mocap_data_file.endswith(".fbx") or test_mocap_data_file.endswith(".FBX"):
    fbx_data = fbx_tools.load(test_mocap_data_file)
    test_mocap_data = mocap_tools.fbx_to_mocap(fbx_data)[0] # first skeleton only  
    
test_mocap_data["skeleton"]["offsets"] *= mocap_pos_scale
test_mocap_data["motion"]["pos_local"] *= mocap_pos_scale

# set x and z offset of root joint to zero
test_mocap_data["skeleton"]["offsets"][0, 0] = 0.0 
test_mocap_data["skeleton"]["offsets"][0, 2] = 0.0 

if test_mocap_data_file.endswith(".bvh") or test_mocap_data_file.endswith(".BVH"):
    test_mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat_bvh(test_mocap_data["motion"]["rot_local_euler"], test_mocap_data["rot_sequence"])
elif test_mocap_data_file.endswith(".fbx") or test_mocap_data_file.endswith(".FBX"):
    test_mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat(test_mocap_data["motion"]["rot_local_euler"], test_mocap_data["rot_sequence"])

test_mocap_data = torch.from_numpy(test_mocap_data["motion"]["rot_local"]).to(torch.float32)
test_mocap_data = test_mocap_data.reshape(-1, pose_dim)


print("test_mocap_data s ", test_mocap_data.shape)

test_mocap_start_times = [100, 200, 300, 400]
test_mocap_duration = 10
    
for test_mocap_start_time in test_mocap_start_times:
    create_ref_anim(test_mocap_data, test_mocap_start_time * mocap_fps, 10 * mocap_fps, "results/anims/test_mocap_{}.gif".format(test_mocap_start_time))
    create_pred_audio3(test_mocap_data, test_audio_data, test_mocap_start_time * mocap_fps, 10 * mocap_fps, "results/audio/gen_audio_{}-{}_epoch_{}_test.wav".format(test_mocap_start_time, (test_mocap_start_time + test_mocap_duration), epochs))

