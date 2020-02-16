import pandas as pd
import numpy as np
import librosa
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error

from scipy.stats import mode


class TreeNode:
    def __init__(self, left=None, right=None, split_fn=None, leaf_evaluate=None):
        self.left = left
        self.right = right
        self.split_fn = split_fn
        self.leaf_evaluate = leaf_evaluate

    def is_leaf(self):
        return self.left == None and self.right == None

    def evaluate(self, X_i):
        if self.is_leaf():
            return self.leaf_evaluate()
        if self.split_fn(X_i):
            return self.left.evaluate(X_i)
        else:
            return self.right.evaluate(X_i)


class Leaf(TreeNode):

    def __init__(self, label):
        TreeNode.__init__(self, leaf_evaluate=lambda: label)

def H(y):
    def proportion(val, y):
        return (y == val).sum() / len(y)
    unique = set(y)
    return sum(-1 * proportion(val, y) * np.log2(proportion(val, y)) for val in unique)
def weighted_entropy(yes, no):
    total_size = len(yes) + len(no)
    return (len(yes) / total_size) * H(yes) + (len(no) / total_size) * H(no)

def train(X_train, Y_train, max_depth=None):
    if len(Y_train) == 0:
        return Leaf(0)

    if len(set(Y_train)) == 1 or max_depth == 1:
        return Leaf(mode(Y_train).mode)

    def split_weighted_entropy(feature_idx, feature_value):
        """Calculate the weighted entropy of the split on feature <feature_idx>,
        and on value <feature_value> """
        feature = X_train[:, feature_idx]
        yes = Y_train[feature > feature_value]
        no = Y_train[feature <= feature_value]
        return weighted_entropy(yes, no)
    splits = np.zeros(X_train.shape)
    for feature_idx in range(X_train.shape[1]):
        # try to split on each X-value, no reason to try others
        for i, feature_value in enumerate(X_train[:, feature_idx]):
            splits[i, feature_idx] = split_weighted_entropy(
                feature_idx, feature_value)

    # find best split
    max_idxs = X_train.argmax(axis=0)
    for col, max_idx in enumerate(max_idxs):
        splits[max_idx, col] = float('inf')

    i = np.argmin(splits)
    best_feature_idx = i % splits.shape[1]
    best_feature_value = X_train[i // splits.shape[1], best_feature_idx]

    yes = X_train[:, best_feature_idx] > best_feature_value
    no = X_train[:, best_feature_idx] <= best_feature_value

    # recurse and make decision trees on the yes and no sets
    tree = TreeNode(
        split_fn=lambda X_i: X_i[best_feature_idx] > best_feature_value,
        left=train(X_train[yes], Y_train[yes], max_depth=max_depth -
                   1 if max_depth is not None else None),
        right=train(X_train[no], Y_train[no], max_depth=max_depth -
                    1 if max_depth is not None else None)
    )
    return tree

def accuracy(y_pred, y_true):
    return (y_pred == y_true).sum() / y_true.shape[0]


def predict(X, tree):
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    preds = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        preds[i] = tree.evaluate(X[i])
    return preds


def parser(label, fileName):
    moodDic = {'belly_pain': 0, 'burping': 1,
               'discomfort': 2, "hungry": 3, "tired": 4}
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
    
    feature = [moodDic[label]] + list(feature)
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

X = sound.iloc[:, 1:].values
Y = sound.iloc[:, 0].values

X_train_sound, X_test_sound, Y_train_sound, Y_test_sound = train_test_split(
    X, Y, test_size=0.2, random_state=0)
print(sound)
print(len(X_train_sound), len(Y_train_sound))    
tree = train(X_train_sound, Y_train_sound)
preds = predict(X_test_sound, tree)

print(accuracy(preds, Y_test_sound))


