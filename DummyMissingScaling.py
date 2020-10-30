# dummy variables

X_train_dummy = pd.get_dummies(X_train2)
X_val_dummy = pd.get_dummies(X_val2)
X_test_dummy = pd.get_dummies(X_test2)


# missing values in features

from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imp.fit(X_train_dummy)
X_train_after_missing = imp.transform(X_train_dummy)
X_val_after_missing = imp.transform(X_val_dummy)
X_test_after_missing = imp.transform(X_test_dummy)
#print(X_train_after_missing.shape)

col_dummy = ['Pclass', 'Age', 'familySize', 'Fare',
             'smallFamily', 'mediumFamily', 'largeFamily',
             'minor1', 'SurnameAllDied' , 'SurnameOneLived',
             'Sex',  'Sex_male']

X_train_Amissing = pd.DataFrame(X_train_after_missing,columns=col_dummy)
X_val_Amissing = pd.DataFrame(X_val_after_missing,columns=col_dummy)
X_test_Amissing = pd.DataFrame(X_test_after_missing,columns=col_dummy)

X_train_copy_n = X_train_Amissing.copy()
X_val_copy_n = X_val_Amissing.copy()
X_test_copy_n = X_test_Amissing.copy()


X_train_Amissing=X_train_copy_n.drop(columns=['Sex_male'])
X_val_Amissing=X_val_copy_n.drop(columns=['Sex_male'])
X_test_Amissing = X_test_copy_n.drop(columns=['Sex_male'])

X_train_Amissing.head(5)


# feature scaling

from sklearn.preprocessing import StandardScaler
X_train_copy = X_train_Amissing.copy()
X_val_copy = X_val_Amissing.copy()
X_test_copy = X_test_Amissing.copy()

col_scaling = ['Age', 'familySize','Fare']
X_train_sc = X_train_copy[col_scaling]
X_val_sc = X_val_copy[col_scaling]
X_test_sc = X_test_copy[col_scaling]
sc = StandardScaler()
sc.fit(X_train_sc.values)
X_train_sc = sc.transform(X_train_sc.values)
X_val_sc = sc.transform(X_val_sc)
X_test_sc = sc.transform(X_test_sc)

X_train_Amissing[col_scaling] = X_train_sc
X_val_Amissing[col_scaling] = X_val_sc
X_test_Amissing[col_scaling] = X_test_sc

X_train_Amissing.head(5)

X_train_final = X_train_Amissing
X_val_final = X_val_Amissing
X_test_final = X_test_Amissing
