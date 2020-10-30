# DNN model: accuracy = 0.80382

import tensorflow as tf 
from tensorflow.keras import regularizers
model = tf.keras.models.Sequential([tf.keras.layers.Dense(20, activation=tf.nn.relu),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(15, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.01)),
                                    tf.keras.layers.Dropout(0.1),
                                    tf.keras.layers.Dense(10, activation=tf.nn.relu),
                                    tf.keras.layers.Dropout(0.1),
                                    tf.keras.layers.Dense(10, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.01)),
                                    tf.keras.layers.Dropout(0.1),
                                    tf.keras.layers.Dense(10, activation=tf.nn.relu),
                                    tf.keras.layers.Dropout(0.1),
                                    tf.keras.layers.Dense(10, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.01)),
                                    tf.keras.layers.Dropout(0.1),
                                    tf.keras.layers.Dense(10, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.01)),
                                    tf.keras.layers.Dropout(0.1),
                                    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)])

opt = tf.keras.optimizers.Adam(lr=1e-3)

model.compile(optimizer = opt,
              loss = 'binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_final,  np.asarray(y_train_final), epochs=200)


model.summary()
y_pred = model.predict_classes(X_val_final)


# confusion_matrix and  accuracy_score
cm2 = confusion_matrix(y_val_final, np.array(y_pred))
print('confusion matrix = \n', cm2)
ac = accuracy_score(y_val_final, y_pred)
print('accuracy score = ', ac)

# predicting the test set
predictions = model.predict_classes(X_test_final)
