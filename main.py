import tensorflow as tf
from tensorflow import keras
import os

content_dir = '/Users/kashyapbrahmandam/Desktop/Kaggle Files/content'
checkpoint_dir='Model Check Points/Best-Model.h5'
checkpoint_path=os.path.join(content_dir,checkpoint_dir)

def model(x_train,y_train,x_val,y_val,outshape=6):

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv1D(128, 10, padding='same', input_shape=x_train.shape[1:]))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=(4)))

    model.add(tf.keras.layers.Conv1D(64, 10, padding='same' ))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=(2)))

    model.add(tf.keras.layers.GlobalAveragePooling1D())

    model.add(tf.keras.layers.Dense(256, activation='relu',kernel_regularizer='l2'))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(128, activation='relu',kernel_regularizer='l2'))
    model.add(tf.keras.layers.Dropout(0.1))

    model.add(keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(outshape))
    model.add(tf.keras.layers.Activation('softmax'))

    # set callbacks
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                     factor=0.5, patience=4,
                                                     verbose=1, mode='max',
                                                     min_lr=0.00001)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=42,
                                                  verbose=1)

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                          save_weights_only=False,
                                                          monitor='val_accuracy',
                                                          mode='max',
                                                          save_best_only=True)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=256, epochs=100,
                        callbacks=[early_stop, model_checkpoint, reduce_lr], verbose=2)

    return history,model

def get_loss_acc(model,x_val,y_val):
    loss, acc = model.evaluate(x_val, y_val, verbose=2)
    print('Model Accuracy: {:5.2f}%'.format(100 * acc))
    return loss,acc




