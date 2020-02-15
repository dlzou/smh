import pandas as pd
import librosa
import glob
import os


def parser(fileName):
    # function to load files and extract features
    file_name = fileName;

    # handle exception to check if there isn't a file which is corrupted
    try:
        # here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(fileName, res_type='kaiser_fast')
        # we extract mfcc feature from data
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        feature = mfccs
    except Exception as e:
        print("Error encountered while parsing file: ", fileName)
        return None, None
    
    feature = mfccs
    return feature


def getFiles(mood):
    filePath = os.path.join("./rawData", str(mood))
    files = list()
    for i in os.listdir(filePath):
        if i.endswith('.wav'):
            files.append(i)
    return files

fullData = {}
for j in os.listdir("rawData"):
    fList = getFiles(str(j))
    fullData[j] = fList
print(fullData)
#temp = train.apply(parser, axis=1)
#temp.columns = ['feature', 'label']

