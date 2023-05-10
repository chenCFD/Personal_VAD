

import soundfile as sf
from glob import glob
import os
from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import librosa
import torch
import random
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import math
import matplotlib.pyplot as plt




def make_dvector(file_loc):
    dvector_model = VoiceEncoder()

    wavs_dvector = []

    x0, sr = sf.read(file_loc)
    x=remove_silence(x0)
        
    wavs_dvector.append(preprocess_wav(x))
        
    dvector = dvector_model.embed_speaker(wavs_dvector)
    return dvector

def make_mel_feature(file_loc):
    x, sr = sf.read(file_loc)
    arr0 = x.astype(np.float32, order='C')
    arr=remove_silence(arr0)
    
    # extract the filterbank features
    fbanks = librosa.feature.melspectrogram(arr, 16000, n_fft=640,
            hop_length=320, n_mels=40).astype('float32').T[:-2]
    logfbanks = np.log10(fbanks + 1e-6)
    return logfbanks

def combine_feature(logfbanks,dvector):
    x1 = np.hstack((logfbanks, np.full((logfbanks.shape[0], 256), dvector))) 
    return x1

def dbfs(sound):
    dbfs=20*math.log10(np.mean(sound**2)+1e-15)
    return dbfs

def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0 # ms

    assert chunk_size > 0 # to avoid infinite loop
    while dbfs(sound[trim_ms:trim_ms+chunk_size]) < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms

def remove_silence(sound):
    start_trim = detect_leading_silence(sound)
    end_trim = detect_leading_silence(sound[::-1])

    duration = len(sound)    
    trimmed_sound = sound[start_trim:duration-end_trim]
    return trimmed_sound


def load_data(path,firstfile,downfile,sr,p):
    
    #create different length ns
    ns=[]
    for k in range(3):
        arr=np.zeros(int(sr*1.5*(k+1)),dtype='float32')
        ns.append(arr)
    
    
    #paper: 300000 concatenated for train and 5000 concatenated for test.
    #concat audio and create mel spec
    
    
    count=0
    
    
    while p<len(downfile)-1:
        
    
        for t in range(len(downfile[p])):
            
            #select target and non-target
            pathst=path+'/'+firstfile[p]+'/'+downfile[p][t]
            filest = glob(os.path.join(pathst,"*.flac"))
            
            pn=random.randint(0, len(downfile)-1)
            
            
            if pn==p:
                if len(downfile[p+1])>t:
                    
                    pathsnt=path+'/'+firstfile[p+1]+'/'+downfile[p+1][t]
                    filesnt = glob(os.path.join(pathsnt,"*.flac"))
                else:
                    pathsnt=path+'/'+firstfile[p+1]+'/'+downfile[p+1][0]
                    filesnt = glob(os.path.join(pathsnt,"*.flac"))
                    
            else:
                if len(downfile[pn])>t:
                    
                    pathsnt=path+'/'+firstfile[pn]+'/'+downfile[pn][t]
                    filesnt = glob(os.path.join(pathsnt,"*.flac"))
                else:
                    pathsnt=path+'/'+firstfile[pn]+'/'+downfile[pn][0]
                    filesnt = glob(os.path.join(pathsnt,"*.flac"))
                
            for st in range(len(filest)-1):
                if st%2==0:
                    dpath=filest[st]
                    dvector=make_dvector(dpath)
                    
                    path_tss=filest[st+1]
                    tss_feature=make_mel_feature(path_tss)
                    
                    tss_label=np.ones(len(tss_feature[:,0]),dtype='float32')*2
                    
                    
                    path_ntss=filesnt[st%len(filesnt)]
                    ntss_feature=make_mel_feature(path_ntss)
                    
                    ntss_label=np.ones(len(ntss_feature[:,0]),dtype='float32')
                
                else:
                    dpath=filesnt[st%len(filesnt)]
                    dvector=make_dvector(dpath)
                    
                    path_tss=filesnt[(st+1)%len(filesnt)]
                    tss_feature=make_mel_feature(path_tss)
                    
                    tss_label=np.ones(len(tss_feature[:,0]),dtype='float32')*2
                    
                    
                    path_ntss=filest[st]
                    ntss_feature=make_mel_feature(path_ntss)
                    
                    ntss_label=np.ones(len(ntss_feature[:,0]),dtype='float32')
                
                
                nss=ns[st%len(ns)]
                nssfbanks = librosa.feature.melspectrogram(nss, 16000, n_fft=640,
                        hop_length=320, n_mels=40).astype('float32').T[:-2]
                nss_feature = np.log10(nssfbanks + 1e-6)
                    
                nss_label=np.zeros(len(nss_feature[:,0]),dtype='float32')
                    
                if (st==0 or st==1) and t==0 and p==0:
                    btss, sr = sf.read(dpath)
                    bedtss = btss.astype(np.float32, order='C')
                    bedtss=remove_silence(bedtss)
                        
                    xtss, sr = sf.read(path_tss)
                    arrtss = xtss.astype(np.float32, order='C')
                    arrtss=remove_silence(arrtss)
                        
                    xntss, sr = sf.read(path_ntss)
                    arrntss = xntss.astype(np.float32, order='C')
                    arrntss=remove_silence(arrntss)
                        
                    test1=np.concatenate((arrtss,arrntss,nss),axis=0)
                    sf.write('outputtest_'+str(st)+'.flac' , test1, sr)
                    sf.write('outputembedding_'+str(st)+'.flac' , bedtss, sr)
                
                concat=[]
                concat.append([tss_feature,tss_label])
                concat.append([ntss_feature,ntss_label])
                concat.append([nss_feature,nss_label])
                random.shuffle(concat)
                
                
                features=np.concatenate((concat[0][0],concat[1][0],concat[2][0]),axis=0)
                labels=np.concatenate((concat[0][1],concat[1][1],concat[2][1]),axis=0)
                    
                     
                feature_com=combine_feature(features, dvector)
                    
                
                if st==0 and t==0 and count==0:
                    input_feature=feature_com.T
                    output_label=labels
                    count=count+1
                else:
                    input_feature=np.hstack((input_feature,feature_com.T))
                    output_label=np.hstack((output_label,labels))
        # print(input_feature.shape[1])
        if input_feature.shape[1]>5*64*320:
            break
        else:
            p=p+1
        

    
    output_label = output_label.astype(int)
    output_label = np.eye(3)[output_label]
    
    input_feature=input_feature.T
    
    total_frame=np.size(input_feature,0)
    seq_length=320
    cut_num=total_frame%seq_length
    
    input_feature_c=input_feature[:total_frame-cut_num,:]
    output_label_c=output_label[:total_frame-cut_num,:]
    
    num_seq=int((total_frame-cut_num)/seq_length)
    
    input_feature_r=input_feature_c.reshape((num_seq,seq_length,np.size(input_feature,1)))   #[num_seq,320,296]
    output_label_r=output_label_c.reshape((num_seq,seq_length,np.size(output_label,1)))    #[num_seq,320,3]
    
                    
    tensor_x = torch.Tensor(input_feature_r) # transform to torch tensor
    tensor_y = torch.Tensor(output_label_r)
    
    my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    
    train_size = int(0.9 * len(my_dataset))
    test_size = len(my_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(my_dataset, [train_size, test_size])
    
    
    my_dataloader_train = DataLoader(train_dataset,batch_size=64,shuffle=True) # create your dataloader     
    my_dataloader_test = DataLoader(test_dataset,batch_size=64,shuffle=False)

    return my_dataloader_train,my_dataloader_test,p          
                       


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
        output,_ = self.lstm(x)
        return self.fc(output)

def validation(model,my_dataloader_test):
    model.eval()
    total_loss=0
    num_data=0
    with torch.no_grad():
        for x,y in my_dataloader_test:
            x=x.to(device)
            y=y.to(device)
            
            pred=model(x)
            
            loss_fun=nn.CrossEntropyLoss()
            loss=loss_fun(pred,y)
            
            total_loss+=loss
            num_data+=len(x)
            
    loss_val=total_loss/num_data
    print(f'val loss:{loss_val}')
    return loss_val
            



def train(num_epochs,optimizer,model,my_dataloader_train,my_dataloader_test,val_loss_count,stop_val_count):
    

    for epoch in range(num_epochs):
        print(f"====== Starting epoch {epoch} ======")
        for batch, (x,y) in enumerate(my_dataloader_train):
            x=x.to(device)
            y=y.to(device)
            model.train()
            
            pred=model(x)
            
            loss_fun=nn.CrossEntropyLoss()
            loss=loss_fun(pred,y)
            
            #backprop?
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
        
            if batch % 50 == 0:
                print(f'Batch: {batch}, loss = {loss:.4f}')
            
            if batch % 100 == 0:
                loss_val=validation(model,my_dataloader_test)
                
                if batch==0 and epoch==0:
                    val_loss_record=loss_val
                
                if loss_val < val_loss_record and epoch>20:
                    torch.save(model, './model_'+str(epoch)+'_'+str(batch)+'_'+str(round(loss_val.item(),5))+'.pt')
                    val_loss_record=loss_val
        
        loss_val=validation(model,my_dataloader_test)        
        if stop_val_count==0:
            
            val_loss_count.append(round(loss_val.item(),5))
            
            
    return model,val_loss_count
        


if __name__ == '__main__':
    sr=16000
    
    #read file

    path = 'C:/Users/v003280/Desktop/pvad-try/dev-clean-small'
     

    obj = os.scandir(path)

    firstfile=[]

    for entry in obj :
        if entry.is_dir() or entry.is_file():
            firstfile.append(entry.name)

    downfile=[]
    for i in range(len(firstfile)):
        path1=path+'/'+firstfile[i]
        obj1 = os.scandir(path1)

        firstfile1=[]

        for entry1 in obj1 :
            if entry1.is_dir() or entry1.is_file():
                firstfile1.append(entry1.name)
            
        downfile.append(firstfile1)
        
    USE_GPU=True
    if USE_GPU and torch.cuda.is_available():
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')
    
    model=personal_vad_et(296,64,2,3).cpu()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs=3

    val_loss_count=[]
    
    p=0
    
    stop_val_count=0
    
    train_count=0
    
    while p<len(downfile)-1:
        my_dataloader_train,my_dataloader_test,p=load_data(path,firstfile,downfile,sr,p)
        if p>=len(downfile)-1:
            stop_val_count=1
        model,val_loss_count=train(num_epochs,optimizer,model,my_dataloader_train,my_dataloader_test,val_loss_count,stop_val_count)
        
        torch.save(model, './model_'+str(p)+'.pt')
        
        train_count+=1
        

    #save model

    torch.save(model, './model.pt')


    t = np.linspace(1,len(val_loss_count), len(val_loss_count))
    plt.plot(t,val_loss_count, "b-")
    plt.show()
    
    
