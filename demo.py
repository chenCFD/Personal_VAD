# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 09:46:39 2023

@author: V003280
"""

import numpy as np

import torch


import os

from glob import glob




from resemblyzer import VoiceEncoder, preprocess_wav
import soundfile as sf
import librosa

from sklearn import preprocessing
import matplotlib.pyplot as plt
import torch.nn as nn

#model

class personal_vad_et(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, out_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc=nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        print(x.shape)
        output,_ = self.lstm(x)
        return self.fc(output)

test_dir = 'data/eval_dir/'
data = 'data/test'

# move to the evaluation directory
os.chdir(test_dir)

# get the model list
models = glob('models/*pt')

# evaluate the models one by one...
for model in models:
    

    # load the model
    #net = personal_vad_et(input_dim=296, hidden_dim=64, num_layers=2,out_dim=3)
    net=torch.load(model,map_location=torch.device('cpu'))
    
    
    
    
    #create embeddings
    
    dvector_model = VoiceEncoder()

    file = glob(os.path.join("embedding","*.flac"))

    wavs_dvector = []

    x, sr = sf.read(file[0])
        
    wavs_dvector.append(preprocess_wav(x))
        
    dvector = dvector_model.embed_speaker(wavs_dvector)
    
    
    #create features
    
    file = glob(os.path.join("test","*.flac"))

    x, sr = sf.read(file[0])
    arr = x.astype(np.float32, order='C')
    #for p in range(len(arr)-sr*4):
    #    arr[sr*4+p]=arr[sr*4+p]*2


    # extract the filterbank features
    fbanks = librosa.feature.melspectrogram(arr, 16000, n_fft=640,
            hop_length=320, n_mels=40).astype('float32').T[:-2]
    logfbanks = np.log10(fbanks + 1e-6)
    
    
    
    x1 = np.hstack((logfbanks, np.full((logfbanks.shape[0], 256), dvector)))
    x1 = x1.reshape(( 1,x1.shape[0],x1.shape[1]))

    x = torch.from_numpy(x1).float()
    print(x.shape)
    
   

    # set the device to cuda and move the model to the gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net = net.to(device)

    # mAP evaluation
    softmax = torch.nn.Softmax(dim=1)
    
    outputs = []

    with torch.no_grad():
        
        
        

        # pass the data through the model
        netoutput = net(x.to(device))
      
        p = softmax(netoutput[0])    #(1,~,3)to(~,3)
        outputs.append(p.cpu().numpy())
    
    output = np.concatenate(outputs)

    
    tss=output[:,2]
    t=np.linspace(0, len(arr)/sr,len(tss))
    plt.plot(t,tss)
    plt.title(model)
    plt.show()
    
    
    for i in range(len(tss)):
        if tss[i]>0.8:
            tss[i]=1
        else:
            tss[i]=0
    plt.plot(t,tss)
    plt.show()
    
    t1=np.linspace(0, len(arr)/sr,len(arr))
    plt.plot(t1,arr)
    plt.show()
    
    for j in range(len(arr)):
        
        k=int(j*len(tss)/len(arr))
        arr[j]=arr[j]*tss[k]
        
    
    plt.plot(t1,arr)
    plt.show()
    
    sf.write('output_from'+model[8:-3]+'.flac' , arr, sr)
    


