import pandas as pd
import numpy as np
import librosa
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error


def evaluate(y_pred, y_true):
    return np.sqrt(mean_squared_log_error(y_pred, y_true))


def parser(label, fileName):
    # function to load files and extract features
    file_name = os.path.join('./rawData/'+ str(label) + '/' + str(fileName))
    # handle exception to check if there isn't a file which is corrupted
    try:
        # here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        # we extract mfcc feature from data
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        feature = mfccs
    except Exception as e:
        print("Error encountered while parsing file: ", fileName)
        return []
    
    feature = [label] + list(feature)
    return feature


def getFiles(mood):
    filePath = os.path.join("./rawData", str(mood))
    files = list()
    for i in os.listdir(filePath):
        if i.endswith('.wav'):
            files.append(i)
    return files

fullData = []
for j in os.listdir("rawData"):
    fList = getFiles(str(j))
    fullData.append((j,fList))
baby = pd.DataFrame(fullData, columns = ['mood', 'file']).set_index('mood')

#curFile = baby.loc['belly_pain']['file'][0]

#x1 = parser('belly_pain', '549a46d8-9c84-430e-ade8-97eae2bef787-1430130772174-1.7-m-48-bp.wav')
#print(x1)


fileNum = 0
soundDic = {}
for label in baby.index:
    for lst in baby.loc[label]:
        for soundFile in lst:      
           features = parser(label, soundFile)
           soundDic[fileNum] = features
           fileNum += 1

sound = pd.DataFrame.from_dict(data = soundDic, orient = 'index',columns=['mood'] + list(range(0, 40)))
print(sound)
X = sound.iloc[:, 1:].values
Y = sound.iloc[:, 0].values

X_train_sound, X_test_sound, Y_train_sound, Y_test_sound = train_test_split(
    X, Y, test_size=0.2, random_state=0)

print(len(X_train_sound), len(Y_train_sound))    
linear_model = LinearRegression(fit_intercept=True)

linear_model = linear_model.fit(X_train_sound, Y_train_sound)
y_val_pred = linear_model.predict(X_test_sound)
y_val_pred = y_val_pred.clip(min=0)
print(evaluate(y_val_pred, Y_test_sound))

#temp = train.apply(parser, axis=1)
#temp.columns = ['feature', 'label']

