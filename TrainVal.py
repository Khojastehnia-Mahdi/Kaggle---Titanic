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

# creating train and val set

from sklearn.model_selection import train_test_split
X_train, X_val, y_train_final, y_val_final = train_test_split(X_trainVal, y_trainVal, test_size = 0.003)

y_train_np = np.array(y_train_final)

# Feature extraction - step 3
# I created this feature using the insights from the discussion in https://www.kaggle.com/c/titanic/discussion/60946

surnametrain = [i.split(',')[0] for i in X_train.Name]
surnameVal = [i.split(',')[0] for i in X_val.Name]
surnametest = [i.split(',')[0] for i in X_test.Name]

SurnameAllDied_train = np.zeros((len(surnametrain),1))
SurnameOneLived_train = np.zeros((len(surnametrain),1))

SurnameAllDied_val = np.zeros((len(surnameVal),1))
SurnameOneLived_val = np.zeros((len(surnameVal),1))

SurnameAllDied_test = np.zeros((len(surnametest),1))
SurnameOneLived_test = np.zeros((len(surnametest),1))


b=0
for i in surnametrain:
    k=0
    live=0
    dead=0
    d=0
    for j in surnametrain:
        if (i==j and b!=d):
          #  print (i)
            k +=1
            if y_train_np[d] == 1:
                live +=1
            else:
                dead+=1
        d +=1    
    if (dead ==1 and k==1 and live==0) or (dead >=2 and k>=2 and live==0):
        SurnameAllDied_train[b]=1
    if (live>=1 and k>=1):
        SurnameOneLived_train[b]=1
    b+=1
    

b=0
for i in surnameVal:
    k=0
    live=0
    dead=0
    d=0
    for j in surnametrain:
        if i==j:
            k +=1
            if y_train_np[d] == 1:
                live +=1
            else:
                dead+=1
        d +=1   
    if  (dead ==1 and k==1 and live==0) or (dead >=2 and k>=2 and live==0):
        SurnameAllDied_val[b]=1
    if (live>=1 and k>=1):
        SurnameOneLived_val[b]=1
    b+=1
    

b=0
for i in surnametest:
    k=0
    live=0
    dead=0
    d=0
    for j in surnametrain:
        if i==j:
            k +=1
            if y_train_np[d] == 1:
                live +=1
            else:
                dead+=1
        d +=1 
    if  (dead ==1 and k==1 and live==0) or (dead >=2 and k>=2 and live==0):
        SurnameAllDied_test[b]=1
    if (live>=1 and k>=1):
        SurnameOneLived_test[b]=1
    b+=1    
    
    
    # adding the extracted feature

X_train1 = X_train.assign(SurnameAllDied = SurnameAllDied_train)   
X_train2 = X_train1.assign( SurnameOneLived = SurnameOneLived_train) 

X_val1 = X_val.assign(SurnameAllDied = SurnameAllDied_val) 
X_val2 = X_val1.assign( SurnameOneLived = SurnameOneLived_val) 

X_test1 = X_test.assign(SurnameAllDied = SurnameAllDied_test) 
X_test2 = X_test1.assign( SurnameOneLived = SurnameOneLived_test) 

# Removing the feature of 'Name'

X_train2=X_train2.drop(columns=['Name'])
X_val2=X_val2.drop(columns=['Name'])
X_test2 = X_test2.drop(columns=['Name'])
