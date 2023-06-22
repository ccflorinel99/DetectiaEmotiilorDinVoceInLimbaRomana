import librosa
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from datetime import datetime
import os
import random
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.vgg16 import VGG16
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

def write_msg(msg):
    f = open("detalii_antrenare_dataset.txt", "a", encoding="utf-8")
    f.write(msg)
    f.close()
    

file_train_val, file_test, labels_train_val, labels_test = 'spectrograme/spec_train_val.bin', 'spectrograme/spec_test.bin', 'spectrograme/labels_train_val.bin', 'spectrograme/labels_test.bin'
    
class My_label_encoder:
    def __init__(self):
        self.dict = {
            "A" : 0,
            "B" : 1,
            "D" : 2,
            "F" : 3,
            "H" : 4,
            "I" : 5,
            "N" : 6,
            "S" : 7
        }
        
    def encode_list(self, label_list):
        encoded_list = []
        for elem in label_list:
            encoded_list.append(self.dict[elem])
        return encoded_list
    
    def get_no_of_classes(self):
        return len(self.dict)
    
mle = My_label_encoder()
num_classes = mle.get_no_of_classes()

def get_model():
    # Incarca modelul VGG16 preantrenat
    model = VGG16(weights='imagenet', include_top=True)

    # Inlatura layer-ul de output original
    model.layers.pop()

    # Adauga un nou output layer cu num_classes
    output_layer = Dense(num_classes, activation='softmax', name='final_predictions')(model.layers[-1].output)
    model = Model(inputs=model.input, outputs=output_layer)

    # Deblocam greutatile ultimului strat (in caz ca nu e)
    if not model.layers[-1].trainable:
        model.layers[-1].trainable = True
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

    return model

def parse_tfrecord_fn(example_proto):
    feature_description = {
        'value': tf.io.FixedLenFeature([224, 224, 4], tf.float32),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    value = example['value']
    return value

def load_dataset(file_pattern, batch_size=32):
    dataset = tf.data.TFRecordDataset(file_pattern, compression_type='')
    dataset = dataset.map(parse_tfrecord_fn)
    dataset = dataset.batch(batch_size)  # Imparte datele in loturi
    return dataset


batch_size = 10000

filename_model = "model_vgg16_dataset.h5"

if os.path.exists(filename_model):
    model = keras.models.load_model(filename_model)
    model.summary() # verificam arhitectura modelului
else:
    model = get_model()
    model.summary() # verificam arhitectura modelului


class Details:
    def __init__(self):
        self.file = "details_dataset.txt"
        if os.path.exists(self.file):
            with open(self.file, 'r') as f:
                lines = f.readlines()
                if len(lines) != 0:
                    l = lines[0].split(": ")
                    self.epoci = int(l[1].split(",")[0])
                    self.batch_train_val = int(l[2].split(",")[0])
                    self.batch_test = int(l[3])
                else:
                    self.epoci = -1
                    self.batch_train_val = -1
                    self.batch_test = -1
        else:
            self.epoci = -1
            self.batch_train_val = -1
            self.batch_test = -1


    def salveaza(self, epoch, train_val_batch_no, test_batch_no):
        if self.epoci < epoch:
            self.epoci = epoch
            if self.batch_train_val != train_val_batch_no:
                self.batch_train_val = train_val_batch_no
            if self.batch_test != test_batch_no:
                self.batch_test = test_batch_no
        else:
            if self.batch_train_val < train_val_batch_no:
                self.batch_train_val = train_val_batch_no
            if self.batch_test < test_batch_no:
                self.batch_test = test_batch_no
        
            with open(self.file, 'w') as f:
                msg = f"epoci: {self.epoci}, batch_train_val_no: {self.batch_train_val}, batch_test_no: {self.batch_test}"
                f.write(msg)

    def reinit(self):
        self.__init__()

    def get_details(self):
        return self.epoci, self.batch_train_val, self.batch_test

de = Details()
max_epochs = 10

start_all = datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S")
for epoch in range(0, max_epochs):
    done_epochs, trainValBatchNo, testBatchNo = de.get_details()

    if done_epochs <= epoch:
        start_load = datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S")
    
        dataset_train_val = load_dataset(file_pattern=file_train_val, batch_size=batch_size)
        dataset_test = load_dataset(file_pattern=file_test, batch_size=batch_size)
        with open(labels_train_val, 'rb') as f:
            y_train_val_all = np.array(pickle.load(f))

        with open(labels_test, 'rb') as f:
            y_test_all = np.array(pickle.load(f))

        end_load = datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S")
        elapsed = end_load - start_load
        print(f"load elapsed: {elapsed}")

        i = 0
        start_train = datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S")
    
        for batch_train_val in dataset_train_val.as_numpy_iterator():
            if trainValBatchNo < i:
                start, end = i * batch_size, (i + 1) * batch_size
                if end > y_train_val_all.shape[0]:
                    end = -1
                X_train_val = batch_train_val[:, :, :, :3]

                y_train_val = y_train_val_all[start:end]

                if X_train_val.shape[0] > y_train_val.shape[0]:
                    X_train_val = X_train_val[:y_train_val.shape[0]]

                X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42) # 70% date de antrenare, 30% date de testare
    # random_state = 42 means that the split is reproducible, meaning that if the script is run again with the same random_state, the data will be split in the same way
                y_train = to_categorical(y_train, num_classes)
                y_val = to_categorical(y_val, num_classes)

                print(f"X_train.shape = {X_train.shape}, batch no: {i + 1}, epoch no: {epoch + 1}/{max_epochs}")
                print(f"y_train.shape = {y_train.shape}, batch no: {i + 1}, epoch no: {epoch + 1}/{max_epochs}")
                print(f"X_val.shape = {X_val.shape}, batch no: {i + 1}, epoch no: {epoch + 1}/{max_epochs}")
                print(f"y_val.shape = {y_val.shape}, batch no: {i + 1}, epoch no: {epoch + 1}/{max_epochs}")

                model.fit(X_train, y_train, batch_size=15, epochs=10, validation_data=(X_val, y_val))
                de.salveaza(epoch, i, -1)
                model.save(filename_model)
        
            i += 1
        de.salveaza(epoch, i, -1)

        j = 0
        for batch_test in dataset_test.as_numpy_iterator():
            if testBatchNo < j:
                start_test, end_test = j * batch_size, (j + 1) * batch_size

                if end_test > y_test_all.shape[0]:
                    end_test = -1
                X_test = batch_test[:, :, :, :3]
                y_test = to_categorical(y_test_all[start_test:end_test], num_classes)

                if X_test.shape[0] > y_test.shape[0]:
                    X_test = X_test[:y_test.shape[0]]

                print(f"X_test.shape = {X_test.shape}, batch no: {j + 1}, epoch no: {epoch + 1}/{max_epochs}")
                print(f"y_test.shape = {y_test.shape}, batch no: {j + 1}, epoch no: {epoch + 1}/{max_epochs}")

                loss, accuracy = model.evaluate(X_test, y_test)
                print(f"loss: {loss}, accuracy: {accuracy}, batch no: {j + 1}, epoch_no: {epoch + 1}/{max_epochs}")
                write_msg(f"loss: {loss}, accuracy: {accuracy}, batch no: {j + 1}, epoch_no: {epoch + 1}/{max_epochs}\n")
                de.salveaza(epoch, i, j)
                model.save(filename_model)
            j += 1

        model.save(filename_model)
        end_train = datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S")
        print("train end time:", end_train.strftime("%H:%M:%S"), f"epoch no: {epoch + 1}/{max_epochs}")
        elapsed = end_train - start_train
        print(f"elapsed: {elapsed}")
        de.salveaza(epoch + 1, -1, -1)



end_all = datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S")
elapsed = end_all - start_all
print(f"elapsed all: {elapsed}")
write_msg(f"Antrenarea modelului pe baza fisierelor de pe emodb a durat {elapsed}\n")
