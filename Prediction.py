import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from _collections import defaultdict
def StringConvertToInt(String):
    if String=='H':
        return 0
    elif String=='A':
        return 1
    elif String=='D':
        return 2
def ConverToDict(ListH,ListA):
    global DataSet
    ListHDict = defaultdict(list)
    ListADict = defaultdict(list)
    ListH=sorted(ListH,reverse=True)
    ListA=sorted(ListA, reverse=True)
    CheckA=3
    CheckH=3
    CounterH=0
    CounterA = 0
    for value in ListH:
        if len(ListHDict.keys())==3:
            break
        ListHDict[value].append('DataSet')
        CounterH+=1
    for value in ListA:
        if len(ListADict.keys()) == 3:
            break
        ListADict[value].append('DataSet')
        CounterA += 1
    ListH = DataSetPrev.index[DataSetPrev['HomeTeam'] == HomeTeam].tolist()
    ListA = DataSetPrev.index[DataSetPrev['AwayTeam'] == AwayTeam].tolist()
    ListHSEG = DataSetPrevSEG.index[DataSetPrevSEG['HomeTeam'] == HomeTeam].tolist()
    ListASEG = DataSetPrevSEG.index[DataSetPrevSEG['AwayTeam'] == AwayTeam].tolist()
    ListH = sorted(ListH, reverse=True)
    ListA = sorted(ListA, reverse=True)
    index=0
    while CounterH != CheckH and not (CounterH>CheckH):
        if ListH:
            value=ListH[index]
            ListHDict[value].append('DataSetPrev')
        elif ListHSEG:
            value = ListHSEG[index]
            ListHDict[value].append('DataSetPrevSEG')
        CounterH += 1
        index+=1
    index=0
    while CounterA != CheckA and not (CounterA > CheckA):
        if ListA:
            value = ListA[index]
            ListADict[value].append('DataSetPrev')
        elif ListASEG:
            value = ListASEG[index]
            ListADict[value].append('DataSetPrevSEG')
        CounterA += 1
        index += 1
    return ListHDict,ListADict
def Search(Dict,String):
    global DataSet, DataSetPrev, DataSetPrevSEG

    ValuePlus=0
    for index in Dict:
        Data = Dict[index]
        Data = Data[0]
        if Data!='DataSetPrevSEG':
            Data = globals()[Data]
            ValuePlus+=Data.at[index,String]
    return ValuePlus

def SearchRec(Dict,String,RecToCheck):
    global DataSet,DataSetPrev,DataSetPrevSEG
    ValuePlus = 0
    for index in Dict:
        Data=Dict[index]
        Data=Data[0]
        Data=globals()[Data]
        Temp= StringConvertToInt(Data.at[index, String])
        if Temp == StringConvertToInt(RecToCheck):
            ValuePlus+=1
    return ValuePlus

def Insertion(ListH,ListA):
    global DataSet,DataSetToUse
    DataSetToUse.at[i, 'HHG'] = Search(ListH,'FTHG')
    DataSetToUse.at[i, 'HAGC'] = Search(ListH,'FTAG')
    DataSetToUse.at[i, 'AAG'] = Search(ListA,'FTAG')
    DataSetToUse.at[i, 'AHGC'] = Search(ListA,'FTHG')


    DataSetToUse.at[i, 'HHW'] = SearchRec(ListH, 'FTR', 'H',)
    DataSetToUse.at[i, 'HHD'] = SearchRec(ListH, 'FTR', 'D',)
    DataSetToUse.at[i, 'HHL'] = SearchRec(ListH, 'FTR', 'A',)
    DataSetToUse.at[i, 'AAW'] = SearchRec(ListA, 'FTR', 'A',)
    DataSetToUse.at[i, 'AAD'] = SearchRec(ListA, 'FTR', 'D',)
    DataSetToUse.at[i, 'AAL'] = SearchRec(ListA, 'FTR', 'H',)

    DataSetToUse.at[i, 'HHGH'] = Search(ListH,'HTHG',)
    DataSetToUse.at[i, 'HAGCH'] = Search(ListH,'HTAG',)
    DataSetToUse.at[i, 'AAGH'] = Search(ListA,'HTAG',)
    DataSetToUse.at[i, 'AHGCH'] = Search(ListA,'HTHG',)

    DataSetToUse.at[i, 'HHS'] = Search(ListH,'HS')
    DataSetToUse.at[i, 'HHST'] = Search(ListH,'HST')
    DataSetToUse.at[i, 'AAS'] = Search(ListA,'AS')
    DataSetToUse.at[i, 'AAST'] = Search(ListA,'AST')

    #if len(ListH)>3 and len(ListA)>3:
#pd.set_option('display.max_columns', None)
DataSet = pd.read_csv('CurrentSeason.csv')
DataSetPrev = pd.read_csv('PrevSeason.csv')
DataSetPrevSEG = pd.read_csv('PrevSeason.csv')
DataSet=DataSet[DataSet.columns[0:22]]
DataSetToUse=pd.DataFrame(index=DataSet.index)
DataSetToUse['HomeTeam']=DataSet['HomeTeam']
DataSetToUse['AwayTeam']=DataSet['AwayTeam']
DataSetToUse['FTR']=DataSet['FTR']
Counter=0
for i in range(0,len(DataSetToUse)):
    HomeTeam=DataSetToUse.at[i, 'HomeTeam']
    AwayTeam=DataSetToUse.at[i, 'AwayTeam']
    ListH=DataSet[:i].index[DataSet[:i]['HomeTeam'] == HomeTeam].tolist()
    ListA=DataSet[:i].index[DataSet[:i]['AwayTeam'] == AwayTeam].tolist()
    #ListHMix=DataSet[:i].index[(DataSet[:i]['HomeTeam'] == HomeTeam) | (DataSet[:i]['AwayTeam'] == HomeTeam)].tolist()
    #ListAMix=DataSet[:i].index[(DataSet[:i]['AwayTeam'] == AwayTeam) | (DataSet[:i]['HomeTeam'] == AwayTeam)].tolist()
    ListH,ListA=ConverToDict(ListH,ListA)
    Insertion(ListH,ListA)
print(DataSetToUse)
#################### Our DataSet Refining
def GetXandY(DataSetToUse):
    del DataSetToUse['HomeTeam']
    del DataSetToUse['AwayTeam']
    X = DataSetToUse
    y = DataSetToUse['FTR']
    del DataSetToUse['FTR']
    X = DataSetToUse
    return X,y
def GetYinFormat(y):
    Temp = pd.DataFrame(index=y.index)
    Temp['FTR'] = -1
    count = 0
    for i in y:
        if i == 'H':
            Temp['FTR'][count] = 0
        elif i == 'A':
            Temp['FTR'][count] = 1
        elif i == 'D':
            Temp['FTR'][count] = 2
        count += 1
    y = Temp
    return y

X,y=GetXandY(DataSetToUse)
y=GetYinFormat(y)

####################### XGBoost #################################
import xgboost as xgb
data_dmatrix = xgb.DMatrix(data=X,label=y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
param = {
    'max_depth': 2,  # the maximum depth of each tree
    'eta': 0.2,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 3,  # the number of classes that exist in this datset
    'n_estimators' : 50,
    'learning_rate':0.01,
    'early_stopping_rounds':1,
    }

num_round = 7  # the number of training iterations
bst = xgb.train(param, dtrain, num_round)
preds = bst.predict(dtest)
import numpy as np
best_preds = np.asarray([np.argmax(line) for line in preds])
accuracy = accuracy_score(y_test, best_preds)
print("XGBoost Accuracy :" , (accuracy * 100.0))
#print("XGBoost Confusion Matrix :" )
#print(confusion_matrix(y_test, best_preds))


#################### SVC ############################

from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf',C=1.0)
svclassifier.fit(X_train, y_train.values.ravel())
y_pred = svclassifier.predict(X_test)
print('SVC Accuracy: ',accuracy_score(y_test,y_pred)*100)
#print("SVC Confusion Matrix :" )
#print(confusion_matrix(y_test, y_pred))

####################
from sklearn.ensemble import AdaBoostClassifier
svc=SVC(probability=True, kernel='rbf',C=10)
abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1,base_estimator=svc)
# Train Adaboost Classifer
model = abc.fit(X_train, y_train.values.ravel())
#Predict the response for test dataset
y_pred = model.predict(X_test)
print('AdaBoost Accuracy: ',accuracy_score(y_test,y_pred)*100)
#print("AdaBoost Confusion Matrix :" )
#print(confusion_matrix(y_test, y_pred))

#####################
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier(  min_samples_leaf=4,
                                     min_samples_split=16, n_estimators=60, n_jobs=3)
model.fit(X_train, y_train.values.ravel())
y_pred = model.predict(X_test)
print('ExtraTrees Accuracy: ',accuracy_score(y_test,y_pred)*100)
#print("ExtraTress Confusion Matrix :" )
#print(confusion_matrix(y_test, y_pred))

####################

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=60,
                               bootstrap = True,
                               max_features = 'sqrt')
model.fit(X_train, y_train.values.ravel())
predictions = model.predict(X_test)
print('RandomForrest Accuracy: ',accuracy_score(y_test,predictions)*100)
#print("RandomForrest Confusion Matrix :" )
#print(confusion_matrix(y_test, predictions))

##############

