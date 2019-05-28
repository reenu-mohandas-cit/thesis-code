#compiling and fitting the baseline models

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os

from glob import glob


def main():

  ((trainX, trainY), (testX, testY)) = tf.keras.datasets.cifar10.load_data()
  print('shape of input:', trainX.shape)
  
  NUM_EPOCHS = 20
  trainX = trainX.astype("uint8")/ 255.0
  testX = testX.astype("uint8")/ 255.0
    
  print("Compiling model...")
  opt = tf.keras.optimizers.SGD(lr=0.01)
  model1 = model_M1()
  model2 = model_M2()
  model3 = model_M3()
  model4 = model_M4()
  model5 = model_M5()
  model6 = model_M6()
  model1.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
  model2.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
  model3.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
  model4.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
  model5.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
  model6.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
   
  print (model1.summary())
    
  print("Training network ModelM1..... ", )
  H1 = model1.fit(trainX, trainY, validation_data=(testX, testY),
                  batch_size=64, epochs=NUM_EPOCHS, validation_split=0.2)
    
  plt.style.use("ggplot")
  plt.figure()
  plt.plot(np.arange(0, NUM_EPOCHS), H1.history["loss"], label="train_loss")
  plt.plot(np.arange(0, NUM_EPOCHS), H1.history["val_loss"], label="val_loss")
  plt.plot(np.arange(0, NUM_EPOCHS), H1.history["acc"], label="train_acc")
  plt.plot(np.arange(0, NUM_EPOCHS), H1.history["val_acc"], label="val_acc")
  plt.title("Training Loss and Accuracy")
  plt.xlabel("Epoch #")
  plt.ylabel("Loss/Accuracy")
  plt.legend()
  plt.show()
  
  print("Training network ModelM2.....")
  H2 = model2.fit(trainX, trainY, validation_data=(testX, testY),
                  batch_size=64, epochs=NUM_EPOCHS, validation_split=0.2)
    
  plt.style.use("ggplot")
  plt.figure()
  plt.plot(np.arange(0, NUM_EPOCHS), H2.history["loss"], label="train_loss")
  plt.plot(np.arange(0, NUM_EPOCHS), H2.history["val_loss"], label="val_loss")
  plt.plot(np.arange(0, NUM_EPOCHS), H2.history["acc"], label="train_acc")
  plt.plot(np.arange(0, NUM_EPOCHS), H2.history["val_acc"], label="val_acc")
  plt.title("Training Loss and Accuracy")
  plt.xlabel("Epoch #")
  plt.ylabel("Loss/Accuracy")
  plt.legend()
  plt.show()
  
  print("Training network ModelM3.....")
  H3 = model3.fit(trainX, trainY, validation_data=(testX, testY),
                  batch_size=64, epochs=NUM_EPOCHS, validation_split=0.2)
    
  plt.style.use("ggplot")
  plt.figure()
  plt.plot(np.arange(0, NUM_EPOCHS), H3.history["loss"], label="train_loss")
  plt.plot(np.arange(0, NUM_EPOCHS), H3.history["val_loss"], label="val_loss")
  plt.plot(np.arange(0, NUM_EPOCHS), H3.history["acc"], label="train_acc")
  plt.plot(np.arange(0, NUM_EPOCHS), H3.history["val_acc"], label="val_acc")
  plt.title("Training Loss and Accuracy")
  plt.xlabel("Epoch #")
  plt.ylabel("Loss/Accuracy")
  plt.legend()
  plt.show()
  
  print("Training network ModelM4.....")
  H4 = model4.fit(trainX, trainY, validation_data=(testX, testY),
                  batch_size=64, epochs=NUM_EPOCHS, validation_split=0.2)
    
  plt.style.use("ggplot")
  plt.figure()
  plt.plot(np.arange(0, NUM_EPOCHS), H4.history["loss"], label="train_loss")
  plt.plot(np.arange(0, NUM_EPOCHS), H4.history["val_loss"], label="val_loss")
  plt.plot(np.arange(0, NUM_EPOCHS), H4.history["acc"], label="train_acc")
  plt.plot(np.arange(0, NUM_EPOCHS), H4.history["val_acc"], label="val_acc")
  plt.title("Training Loss and Accuracy")
  plt.xlabel("Epoch #")
  plt.ylabel("Loss/Accuracy")
  plt.legend()
  plt.show()
  
  print("Training network ModelM5.....")
  H5 = model5.fit(trainX, trainY, validation_data=(testX, testY),
                  batch_size=64, epochs=NUM_EPOCHS, validation_split=0.2)
 
  plt.style.use("ggplot")
  plt.figure()
  plt.plot(np.arange(0, NUM_EPOCHS), H5.history["loss"], label="train_loss")
  plt.plot(np.arange(0, NUM_EPOCHS), H5.history["val_loss"], label="val_loss")
  plt.plot(np.arange(0, NUM_EPOCHS), H5.history["acc"], label="train_acc")
  plt.plot(np.arange(0, NUM_EPOCHS), H5.history["val_acc"], label="val_acc")
  plt.title("Training Loss and Accuracy")
  plt.xlabel("Epoch #")
  plt.ylabel("Loss/Accuracy")
  plt.legend()
  plt.show()
  
  print("Training network ModelM6.....")
  H6 = model6.fit(trainX, trainY, validation_data=(testX, testY),
                  batch_size=64, epochs=NUM_EPOCHS, validation_split=0.2)
    
  plt.style.use("ggplot")
  plt.figure()
  plt.plot(np.arange(0, NUM_EPOCHS), H6.history["loss"], label="train_loss")
  plt.plot(np.arange(0, NUM_EPOCHS), H6.history["val_loss"], label="val_loss")
  plt.plot(np.arange(0, NUM_EPOCHS), H6.history["acc"], label="train_acc")
  plt.plot(np.arange(0, NUM_EPOCHS), H6.history["val_acc"], label="val_acc")
  plt.title("Training Loss and Accuracy")
  plt.xlabel("Epoch #")
  plt.ylabel("Loss/Accuracy")
  plt.legend()
  plt.show()



    
    
    
    
main()
