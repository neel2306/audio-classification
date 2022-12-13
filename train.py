import json
import keras
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATA_PATH = "//home//neelabh//Desktop//Audio_classification//dataset//data.json"
LR = 0.0001
NUM_WORDS = 10
MODEL_PATH = "model.h5"

def data_splits(data_path):
    '''
    Function that splits the dataset into train, test and validation sets.
    param: data_path(str): Path of the dataset.
    '''
    
    #Loading the dataset from json.
    with open(data_path, "r") as f:
        data = json.load(f)
    
    #Extracting inputs -> MFCCs and labels -> labels. Converting them to numpy arrays as well.
    X = np.array(data['MFCCs'])
    y = np.array(data['labels'])
    
    #Splitting the data into train and test splits.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)

    #Splitting the training set again to gain validation sets.
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.1)

    #Converting the inputs from 2d to 3d arrays (to fit into a CNN).
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape, lr):

    model = keras.Sequential()

    #Conv layer 1
    model.add(keras.layers.Conv2D(64, (3,3), activation='relu', input_shape= input_shape, kernel_regularizer=keras.regularizers.l2(0.001))) 
    model.add(keras.layers.BatchNormalization()) 
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding='same'))

    #Conv layer 2
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))) 
    model.add(keras.layers.BatchNormalization()) 
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding='same'))

    #Conv layer 3
    model.add(keras.layers.Conv2D(32, (2,2), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))) 
    model.add(keras.layers.BatchNormalization()) 
    model.add(keras.layers.MaxPool2D((2,2), strides=(2,2), padding='same'))

    #Flattening the 3d output to a 1d array.
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    #Softmax output.
    model.add(keras.layers.Dense(NUM_WORDS, activation='softmax'))

    #Compiling the model.
    optimiser = keras.optimizers.Adam(learning_rate = lr)
    model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model



def main():

    #Loading train, validation and test data.
    X_train, X_validation, X_test, y_train, y_validation, y_test = data_splits(DATA_PATH)

    #Building  the model.
    INPUT_SHAPE = (X_train.shape[1], X_train.shape[2],X_train.shape[3])
    model = build_model(INPUT_SHAPE, LR)

    #Training the model.
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_validation, y_validation))

    #Evaluating the model.
    t_error, t_accuracy = model.evaluate(X_test, y_test)
    print(f"Test error: {t_error}")
    print(f"Test accuracy: {t_accuracy}")

    #Saving the model.
    model.save(MODEL_PATH)

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        print(e)