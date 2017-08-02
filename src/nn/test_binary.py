from sklearn.neural_network import MLPClassifier

if __name__ == '__main__':
    import sys
    sys.path.append('..')

from feature.MFCC import mffcc_from_folder

features, target, userList = mffcc_from_folder('../files/wav')
print(target)
print(features.shape)

def generate_model(features = None, target = None):
    for i in range(len(userList)):
        user = userList[i]
        y_train=[0] * features.shape[0]
        for j in range(len(target)):
            if user == target[j]:
                y_train[j] = 1
        mlp = MLPClassifier(hidden_layer_sizes(50,50,50), activation='logistic')
        mlp.fit(features,y_train)
