import numpy as np 
import pandas as pd

trainVal_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

pd.set_option('display.max_rows', None)
#trainVal_data.head(100)

# Feature Engineering


# Feature extraction - step 1
trainVal_data = trainVal_data .assign(familySize = trainVal_data.Parch.values+trainVal_data.SibSp.values+1)   
test_data = test_data.assign(familySize = test_data.Parch.values+test_data.SibSp.values+1)


trainVal_data['smallFamily'] = trainVal_data['familySize'].map(lambda x: 1 if x<=1 else 0 )
trainVal_data['mediumFamily'] = trainVal_data['familySize'].map(lambda x: 1 if 1<x<=3 else 0 )
trainVal_data['largeFamily'] = trainVal_data['familySize'].map(lambda x: 1 if 3<x else 0 )

test_data['smallFamily'] = test_data['familySize'].map(lambda x: 1 if x<=1 else 0 )
test_data['mediumFamily'] = test_data['familySize'].map(lambda x: 1 if 1<x<=3 else 0 )
test_data['largeFamily'] = test_data['familySize'].map(lambda x: 1 if 3<x else 0 )


trainVal_data1=trainVal_data.drop(columns=['Parch'])
trainVal_data2=trainVal_data1.drop(columns=['SibSp'])

test_data1=test_data.drop(columns=['Parch'])
test_data2=test_data1.drop(columns=['SibSp'])

test_data2.head(5)

# Feature extraction - step 2

y_trainVal=trainVal_data2['Survived']

def newFeature1(X):
    new_f_age =(X.Age <10)
    bb = new_f_age.copy()
    new_name = X.Name
    for i in range(len(new_name)):
          a = new_name[i].find('Master')
          if not a ==-1:
              new_f_age[i] = True
    return new_f_age

new_f_trainval1 = newFeature1(trainVal_data2)        
new_f_test1  = newFeature1(test_data2)


# creating the train/val set and test set. 
# We also will add some features after creating train set and val set.  

X_trainVal = pd.DataFrame({'Name':trainVal_data2.Name, 'Pclass':trainVal_data2.Pclass, 'Sex':trainVal_data2.Sex, 'Age':trainVal_data2.Age,
       'familySize':trainVal_data2.familySize, 'Fare':trainVal_data2.Fare, 
       'smallFamily':trainVal_data2.smallFamily,  'mediumFamily': trainVal_data2.mediumFamily, 
       'largeFamily': trainVal_data2.largeFamily,
        'minor1': new_f_trainval1})


X_test = pd.DataFrame({'Name':test_data2.Name, 'Pclass':test_data2.Pclass, 'Sex':test_data2.Sex, 'Age':test_data2.Age,
       'familySize':test_data2.familySize, 'Fare':test_data2.Fare, 
        'smallFamily':test_data2.smallFamily,  'mediumFamily': test_data2.mediumFamily, 
       'largeFamily':test_data2.largeFamily,
        'minor1': new_f_test1})


X_trainVal.head(5)
