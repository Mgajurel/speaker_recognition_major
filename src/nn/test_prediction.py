import os
import pickle

if __name__ == '__main__':
    import sys
    sys.path.append('..')
from feature.MFCC import mfcc
from nn.neural_network import predict_with_model

modelpath = '../files'
models = [f for f in os.listdir(modelpath) if f.endswith('.pkl')]

test_wave = [f for f in os.listdir(modelpath+'/test/wav') if f.endswith('.wav')]


for model in models:
    mlp = pickle.load(open(modelpath+'/'+ model,'rb'))
    for test_audio in test_wave:
        audiopath = modelpath+'/test/wav' + test_audio
        message = predict_with_model(mlp, audiopath, verbose = True)
