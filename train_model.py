# train_model.py
#By: Sam Schmitz

#    JMZ: This is the simple "very simple script" I use to
#         experiment with simple neural networks. Feel free
#         to modify to your own environment/tastes.

#    Note: This module has a number of global variables some at the
#     top and others are at the bottom in the code that gets the
#     training started that are then used to start training.  It is
#     designed that way to be useful in an interactive terminal
#     session.
#
#     I generally train a model by invoking $ python -i train_model.py
#

import time
import os
import pickle
from pathlib import Path

import numpy as np
from tensorflow import keras
from tensorflow.keras import callbacks, layers, models


# tensorflow and CUDA configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Turn off TF chatter
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Turn off GPU usage
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Turn on GPU usage

# General leraning parameters
# Once tweaked, these tend to stay the same over my trials
LR = .05
BATCH_SIZE = 64
VALIDATION_FRACTION = .05


# I tend to have a number of these build_model type functions to build
# different related models that I am experimenting with.
def build_model():
    input_shape = (10, 10, 6)
    n_outputs = 3
    model = keras.Sequential() 
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    #model.add(layers.Conv2D(256, (2, 2), activation='relu'))
    #model.add(layers.BatchNormalization())
    #model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Dropout(0.5))
    
    model.add(layers.GlobalAveragePooling2D())    
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Dense(n_outputs, activation='softmax'))
    
    opt = keras.optimizers.Adam(learning_rate=LR)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer = opt,
                  metrics = ['accuracy'])
    
    return model


def train(model, train_data, checkpoint_file):
    
    if isinstance(train_data, tuple) and isinstance(train_data[0], tuple):
        (xs, ys), val_data = train_data
    else:
        xs, ys = train_data
        val_data = None

    # Save a "checkpoint" model each time accuracy improves
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=checkpoint_file,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1                  # Print when a model is saved
    )
    # Stop training after 5 epochs without improvement
    early_stopping_cb = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True
    )    
    
    print(" Starting training...")
    model.fit(xs, ys, int(BATCH_SIZE),  
                epochs=25,      #changed to 25
                validation_data=val_data,
                validation_split=0.0 if val_data else VALIDATION_FRACTION,
                callbacks=[checkpoint_cb, early_stopping_cb],
                shuffle=True)
    print("Training complete")


def load_data(path):
    with open(path, "rb") as infile:
        data = pickle.load(infile) 
     
    if isinstance(data, dict) and "train" in data and "val" in data:
        xs_train, ys_train = data["train"]
        xs_val, ys_val = data["val"]
        xs_train = np.array(xs_train, dtype=np.float32)
        ys_train = np.array(ys_train, dtype=np.int8)
        xs_val = np.array(xs_val, dtype=np.float32)
        ys_val = np.array(ys_val, dtype=np.int8)
        return (xs_train, ys_train), (xs_val, ys_val)
    else:
        xs, ys = data
        xs = np.array(xs, dtype=np.float32)
        ys = np.array(ys, dtype=np.int8)        
        return (xs, ys), None            
    


def test(model, test_data_path):
    xs, ys = load_data(test_data_path)
    model.evaluate(xs, ys)   


if __name__ == "__main__":
    # set up files and model structure these 5 configuration lines
    # change from run-to-run during experimentation. The values
    # shown are just for illusration. You can see that I like to
    # have separate subfolders for datasets and models    

    training_data_path = "datasets/ripper_train_image.pkl"
    test_data_path = "datasets/ripper_test_image.pkl"
    checkpoint_path = "models/ripper_image_chk.keras"
    final_model_path = "models/johnny.keras"
    model_build_fn = build_model

    # set up initial model
    yn = "n"
    if Path(checkpoint_path).exists():
        while True:
            yn = input("Continue from existing checkpoint (y/n)? ")
            if yn.lower() in ("y", "n"):
                break
    if yn.lower() == "y":
        model = models.load_model(checkpoint_path)
    else:
        model = model_build_fn()

    model.summary()

    train_data = load_data(training_data_path)
    print("\nTraining", len(train_data[1]))
    start = time.time()    
    train(model, train_data, checkpoint_path)
    end = time.time()
    print("Training finished", round(end-start, 2))

    print("\nRestoring best model")
    model = models.load_model(checkpoint_path)

    print("\nSaving final model:", final_model_path)
    model.save(final_model_path)

    print("\nTesting")
    test(model, test_data_path)
