import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import math

#Separate the data so that only the first three features will be taken into account
#If n is not 3 take all features
def sepData(data,n):
    data_copy = data.copy()
    if n == 3:
        data1 = data_copy.iloc[:,0:3]
    else:
        data1 = data.iloc[:,:]
    return data1

def getData(data,df_class):
#Divide the dataset into three different training and validation sets.
#Validation from 100 to 2000 and remaining as test set in each case
#Since there is no need for optimization in Naive Bayes, dataset is divided into two
    data1 = data.copy()
    data1_train1 = data1[:100]
    data1_val1 = data1[100:].reset_index(drop=True)
    data1_train2 = data1[:1000]
    data1_val2 = data1[1000:].reset_index(drop=True)
    data1_train3 = data1[:2000]
    data1_val3 = data1[2000:].reset_index(drop=True)

    y_train1 = df_class[:100]
    y_train2 = df_class[:1000]
    y_train3 = df_class[:2000]
    y_test1 = df_class[100:].reset_index(drop=True)
    y_test2 = df_class[1000:].reset_index(drop=True)
    y_test3 = df_class[2000:].reset_index(drop=True)
    return data1_train1, data1_val1, data1_train2, data1_val2, data1_train3, data1_val3, y_train1, y_train2, y_train3, y_test1, y_test2, y_test3

#Compute the correct and mislabeled classes
def compute_confusion_matrix(true, pred):

    pred1 = []
    for k in range(pred.shape[0]):
        pred1.append(pred.iloc[k].idxmax())

    K = len(np.unique(true)) # Number of classes
    result = np.zeros((K, K))

    for i in range(len(true)):
        #Decrease 1 from the true label to match the preiction labels
        result[true[i]-1][pred1[i]] += 1

    return result

#Product of all feature columns and the prior prob. of the given class
def colProd(cls,prior):
    res = prior

    for i in range(len(cls)):
        res *= cls[i]

    return res

def calculateGDF(train,test):
    #Empty arrays to keep the GDF results for each class
    gdfYoung = []
    gdfMiddle = []
    gdfOld = []
    #Will be used to filter the train data according to the classes
    young = test[:] == 1
    middle = test[:] == 2
    old = test[:] == 3
    #Prior probabilities of each class
    youngPrior = train[young].shape[0] / train.shape[0]
    middlePrior = train[middle].shape[0] / train.shape[0]
    oldPrior = train[old].shape[0] / train.shape[0]
    #a small correction constant was added to avoid zero variance problem
    corr_factor = 10e-09

    for i in range(train.columns.shape[0]):
        meansYoung = train[:][young].iloc[:,i].mean()
        varYoung = train[:][young].iloc[:,i].var() + corr_factor

        meansMiddle = train[:][middle].iloc[:,i].mean()
        varMiddle = train[:][middle].iloc[:,i].var() + corr_factor

        meansOld = train[:][old].iloc[:,i].mean()
        varOld = train[:][old].iloc[:,i].var() + corr_factor

        meanDiffYoung = (train.iloc[:,i] - meansYoung)**2
        gdfYoung.append((1/math.sqrt(2*math.pi*varYoung)) * np.exp(-1/2*(meanDiffYoung)/varYoung))

        meanDiffMiddle = (train.iloc[:,i] - meansMiddle)**2
        gdfMiddle.append((1/math.sqrt(2*math.pi*varMiddle)) * np.exp(-1/2*(meanDiffMiddle/varMiddle)))

        meanDiffOld = (train.iloc[:,i] - meansOld)**2
        gdfOld.append((1/math.sqrt(2*math.pi*varOld)) * np.exp(-1/2*(meanDiffOld/varOld)))

    probYoung = colProd(gdfYoung,youngPrior)
    probMiddle = colProd(gdfMiddle,middlePrior)
    probOld = colProd(gdfOld,oldPrior)
    #Concatenate the feature table for prediction
    probTable = pd.concat([probYoung, probMiddle, probOld],axis=1)
    coln = np.arange(probTable.shape[1])
    probTable.columns = coln

    return probTable

def gaussNB(train,val,y_train,y_test,title):
    probTrain = calculateGDF(train,y_train)
    probVal = calculateGDF(val,y_test)
    #Suppress the scientific presentation of numbers e.g 1.5 e10-3
    np.set_printoptions(suppress=True)
    trainResult = compute_confusion_matrix(y_train,probTrain)
    valResult = compute_confusion_matrix(y_test,probVal)

    #Column and row labels for the heat map
    df_tr = pd.DataFrame(trainResult, index = [i for i in "123"],
                  columns = [i for i in "123"])
    sn.set(font_scale=0.8)
    plt.figure(figsize = (3,3))
    plt.title("Trn CM for "+title)
    sn.heatmap(df_tr, annot=True, square=True, fmt='g')
    plt.savefig("Trn " + str(title) +".png", format="PNG")

    #Column and row labels for the heat map
    df_val = pd.DataFrame(valResult, index = [i for i in "123"],
                  columns = [i for i in "123"])

    sn.set(font_scale=0.8)
    plt.figure(figsize = (3,3))
    plt.title("Val CM for "+title)
    sn.heatmap(df_val, annot=True, square=True, fmt='g')
    plt.savefig("Val " + str(title) +".png", format="PNG")

    return #trainResult,valResult

def main():
    #Load the data with the column names
    data = pd.read_csv("abalone_dataset.txt", delimiter = "\t", names = ["Sex", "Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscara_weight", "Shell_weight", "Class"])
    #Label encoding for the string features
    data["Sex"].replace({"M": 2, "F": 0, "I": 1}, inplace=True)
    #Copy the class information to new df and drop it from features data
    df_class = data["Class"]
    data.drop(columns="Class",inplace=True)

    data1 = sepData(data,3)
    data1_train1, data1_val1, data1_train2, data1_val2, data1_train3, \
    data1_val3, y_train1, y_train2, y_train3, y_test1, y_test2, y_test3 = getData(data1,df_class)
    #Results for the case: Using only the first 3 features
    gaussNB(data1_train1,data1_val1,y_train1,y_test1,title="3fTrain100ValRest")
    gaussNB(data1_train2,data1_val2,y_train2,y_test2,title="3fTrain1000ValRest")
    gaussNB(data1_train3,data1_val3,y_train3,y_test3,title="3fTrain2000ValRest")

    data2 = sepData(data,8)
    data2_train1, data2_val1, data2_train2, data2_val2, data2_train3, \
    data2_val3, y2_train1, y2_train2, y2_train3, y2_test1, y2_test2, y2_test3 = getData(data2,df_class)
    #Results for the case: Using all features
    gaussNB(data2_train1,data2_val1,y2_train1,y2_test1,title="8fTrain100ValRest")
    gaussNB(data2_train2,data2_val2,y2_train2,y2_test2,title="8fTrain1000ValRest")
    gaussNB(data2_train3,data2_val3,y2_train3,y2_test3,title="8fTrain2000ValRest")

if __name__ == "__main__": main()
