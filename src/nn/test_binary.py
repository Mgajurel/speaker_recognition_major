from sklearn.neural_network import MLPClassifier
import pickle
import os

if __name__ == '__main__':
    import sys
    sys.path.append('..')
    #os.chdir('..')

from feature.MFCC import mffcc_from_folder

def generate_model(features = None, target = None, userList = None):
    for i in range(len(userList)):
        user = userList[i]
        y_train=[0] * features.shape[0]
        for j in range(len(target)):
            if j == target[j]:
                y_train[j] = 1
        mlp = MLPClassifier(hidden_layer_sizes=(50,50,50), activation='logistic')
        mlp.fit(features,y_train)
        pickle.dump(mlp, open('../files/model_'+ user + '.pkl','wb'))


features, target, userList = mffcc_from_folder('../files/wav')
print(target)
print(features.shape)
generate_model(features,target,userList)
