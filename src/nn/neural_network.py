from scipy.io import wavfile
import os
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import decomposition
# import alsaaudio, wave

if __name__ == '__main__':
    import sys
    sys.path.append("..")

from feature.MFCC import mfcc
from feature.MFCC import delta
from feature.MFCC import mffcc_from_folder
from nn.recorder import record_to_file

from feature.sigproc import remove_silence, butter_highpass_filter
from feature.sigproc import silence_zone
import scipy.stats as stats

debug = True
pca = decomposition.PCA(n_components=10)

def print_label(text, character="*"):
    star = int((80-len(text))/2)
    print(character*star, text, character*star)

def show_confusion_matrix(mlp, X_test, y_test, users):

    predictions = mlp.predict(X_test)

    from sklearn.metrics import classification_report,confusion_matrix

    message = "Prediction outcomes"
    message += "\n"
    message += classification_report(y_test, predictions, target_names=users)
    message += "\nconfusion_matrix\n"
    message += np.array_str(confusion_matrix(y_test, predictions))

    return message

def train(folderpath="files/wav", model_name="model.pkl", layer_sizes=(512, 512), verbose=False):
    print("Training started")

    features, target, users = mffcc_from_folder(folderpath)
    print("Features before PCA")
    print("Feature:", features.shape)
    print(features)

    # features = StandardScaler().fit_transform(features)
    print("Calculate PCA of the features")
    pca.fit(features)
    features = pca.transform(features)
    print("Features after PCA")
    print("Feature:", features.shape)
    print(features)

    # Features and target are made now train them
    X_train, X_test, y_train, y_test = train_test_split(features, target,test_size=0.33, random_state=42)

    mlp = MLPClassifier(hidden_layer_sizes=layer_sizes, activation = 'logistic')
    print("Training the model")
    mlp.fit(X_train, y_train)

    message =""
    if verbose:
        message = show_confusion_matrix(mlp, X_test, y_test, users)

    # Training is now complete, save the model
    userList = open("files/metadata.txt", "w")
    for user in users:
        # print(user)
        userList.write(user + '\n')
    userList.close()
    pickle.dump(mlp, open('files/'+model_name,'wb'))
    pickle.dump(pca, open('files/pca.pkl', 'wb'))

    if verbose:
        message += "\n%s is now saved"%model_name
    print("Training complete")

    return message

def predict(model_name="model.pkl", test_path="files/test/wav", verbose=False, threshold=0.95):
    NN = pickle.load(open("files/"+model_name, 'rb'))
    pca = pickle.load(open('files/pca.pkl', 'rb'))
    userList = open("files/metadata.txt", 'r')
    users = userList.read().split('\n')
    userList.close()
    message = ""

    wav_files = [f for f in os.listdir(test_path) if f.endswith('.wav')]

    for audio in wav_files:
        message += "\nThe audio file is %s\n" %audio
        fs, sig = wavfile.read(test_path+'/'+audio)
        filtered_sig = butter_highpass_filter(sig,10,fs)
        filtered_sig = filtered_sig.astype(int)
        # filtered_sig
        filtered_sig = filtered_sig[1000:]

        filtered_sig = remove_silence(fs, filtered_sig)
        mfcc_feat = mfcc(filtered_sig, fs)

        mfcc_feat = pca.transform(mfcc_feat)

        output = NN.predict(mfcc_feat)

        message += get_result_from_output(output, users, verbose, threshold) + "\n"
    return message

def real_time_predict(model_name="model.pkl", speech_len="3", verbose=False, threshold=0.95):
    NN = pickle.load(open("files/"+model_name, 'rb'))
    pca = pickle.load(open('files/pca.pkl', 'rb'))
    userList = open("files/metadata.txt", 'r')
    users = userList.read().split('\n')
    userList.close()
    message = ""

    record_to_file("test.wav", RECORD_SECONDS=speech_len)
    fs, sig = wavfile.read("test.wav")
    filtered_sig = butter_highpass_filter(sig,10,fs)
    filtered_sig = filtered_sig.astype(int)
    # filtered_sig

    filtered_sig = filtered_sig[1000:]

    filtered_sig = remove_silence(fs, filtered_sig)
    feature = mfcc(filtered_sig, fs)
    print("mfcc:",feature.shape)
    feature = pca.transform(feature)
    print("mfcc_pca:",feature.shape)
    output = NN.predict(feature)

    return get_user_from_output(output, users)

def predict_from_file(filename="test.wav"):
    NN = pickle.load(open("files/model.pkl", 'rb'))
    userList = open("files/metadata.txt", 'r')
    users = userList.read().split('\n')
    userList.close()
    message = ""

    fs, sig = wavfile.read(filename)
    filtered_sig = butter_highpass_filter(sig,10,fs)
    filtered_sig = filtered_sig.astype(int)
    # filtered_sig
    filtered_sig = filtered_sig[1000:]

    filtered_sig = remove_silence(fs, filtered_sig)
    mfcc_feat = mfcc(filtered_sig, fs)

    mfcc_feat = pca.transform(mfcc_feat)

    output = NN.predict(mfcc_feat)

    return self.get_user_from_output(output, users)

def get_user_from_output(output, users, threshold=0.95):
    counts = np.bincount(output)
    scores = stats.zscore(counts)
    z_score = scores[np.argmax(counts)]
    z_cdf = stats.norm.cdf(z_score)
    user = users[np.argmax(counts)]
    hcr = counts[np.argmax(counts)]/np.sum(counts)
    print("Counts:",np.array_str(counts))
    print("HMR: %.2f%%" %(hcr*100))
    print("Confidence level: %.2f%%" %(z_cdf*100))
    if z_cdf > threshold:
        return user
    else:
        return "Anynomous"

def get_result_from_output(output, users, verbose=False, threshold=0.95):
    counts = np.bincount(output)
    scores = stats.zscore(counts)
    z_score = scores[np.argmax(counts)]
    z_cdf = stats.norm.cdf(z_score)
    user = users[np.argmax(counts)]
    hcr = counts[np.argmax(counts)]/np.sum(counts)
    message = ""
    if verbose:
        message = "Outcomes"
        message += "\nThe counts are %s" %np.array_str(counts)
        message += "\nConfidence level: %.2f%%" %(z_cdf*100)
        message += "\nThe Highest match ratio: %.2f%%\n" %(hcr*100)

    if z_cdf > threshold:
        message += "The user is: %s\n" %user
    else:
        message += "Sorry, I'm unable to recognize you"

    return message

def record_wav(filename, time=7):
    # record_to_file(filename="test.wav",RECORD_SECONDS=0.5)
    # inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE)
    # inp.setchannels(1)
    # inp.setrate(8000)
    # inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    # inp.setperiodsize(1024)
    # w = wave.open(filename, 'w')
    # w.setnchannels(1)
    # w.setsampwidth(2)
    # w.setframerate(8000)
    #
    # for i in range(time): #~0.5 seconds
    #     l, data = inp.read()
    #     w.writeframes(data)
    #
    # w.close()
    pass

if __name__ == '__main__':
    nn = NeuralNetwork(filepath="../files")
    print_label("Training")
    print(nn.train())
    print_label("Testing from file...")
    print(nn.test_predict())
