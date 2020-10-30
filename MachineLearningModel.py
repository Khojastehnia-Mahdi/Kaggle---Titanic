# train a machine learning model
# SVC algortihm: accuracy = 0.80143

m=2
if m==1:
    from xgboost import XGBClassifier
    model = XGBClassifier()
elif m==2:
    from sklearn.svm import SVC
    model = SVC(kernel = 'rbf')
elif m==3:
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators = 300)
elif m==4:
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p = 2)
elif m==5:
  from sklearn.tree import DecisionTreeClassifier
  model = DecisionTreeClassifier(criterion = "entropy" )

model.fit(X_train_final, y_train_final)

y_pred = model.predict(X_val_final)


# confusion_matrix and  accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_val_final, np.array(y_pred))
print('confusion matrix = \n', cm)
ac = accuracy_score(y_val_final, y_pred)
print('accuracy score = ', ac)

# predicting for the test set
predictions = model.predict(X_test_final)

