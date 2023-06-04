import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
import cv2
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import os
import random
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.vgg16 import VGG16
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def write_msg(msg):
    f = open("detalii_antrenare.txt", "a", encoding="utf-8")
    f.write(msg)
    f.close()
    

path = "emodb/wav"

files, output_file_train_val, output_file_test, output_labels_train_val, output_labels_test = os.listdir(path), 'spectrograme_train_val.bin', 'spectrograme_test.bin', 'labels_train_val.bin', 'labels_test.bin'
    
class My_label_encoder:
    def __init__(self):
        self.dict = {
            "" : 0,
            "A" : 1,
            "A/B" : 2,
            "A/D" : 3,
            "A/F" : 4,
            "A/H" : 5,
            "A/I" : 6,
            "A/N" : 7,
            "A/S" : 8,
            "B" : 9,
            "B/A" : 10,
            "B/D" : 11,
            "B/F" : 12,
            "B/H" : 13,
            "B/I" : 14,
            "B/N" : 15,
            "B/S" : 16,
            "D" : 17,
            "D/A" : 18,
            "D/B" : 19,
            "D/F" : 20,
            "D/H" : 21,
            "D/I" : 22,
            "D/N" : 23,
            "D/S" : 24,
            "F" : 25,
            "F/A" : 26,
            "F/B" : 27,
            "F/D" : 28,
            "F/H" : 29,
            "F/I" : 30,
            "F/N" : 31,
            "F/S" : 32,
            "H" : 33,
            "H/A" : 34,
            "H/B" : 35,
            "H/D" : 36,
            "H/F" : 37,
            "H/I" : 38,
            "H/N" : 39,
            "H/S" : 40,
            "I" : 41,
            "I/A" : 42,
            "I/B" : 43,
            "I/D" : 44,
            "I/F" : 45,
            "I/H" : 46,
            "I/N" : 47,
            "I/S" : 48,
            "N" : 49,
            "N/A" : 50,
            "N/B" : 51,
            "N/D" : 52,
            "N/F" : 53,
            "N/H" : 54,
            "N/I" : 55,
            "N/S" : 56,
            "S" : 57,
            "S/A" : 58,
            "S/B" : 59,
            "S/D" : 60,
            "S/F" : 61,
            "S/H" : 62,
            "S/I" : 63,
            "S/N" : 64
        }
        
    def encode_list(self, label_list):
        encoded_list = []
        for elem in label_list:
            encoded_list.append(self.dict[elem])
        return encoded_list
    
    def get_no_of_classes(self):
        return len(self.dict)
    
mle = My_label_encoder()

# Incarca modelul VGG16 preantrenat
model = VGG16(weights='imagenet', include_top=True)

# Inlatura layer-ul de output original
model.layers.pop()

num_classes = mle.get_no_of_classes()

# Adauga un nou output layer cu num_classes = 64 clase
output_layer = Dense(num_classes, activation='softmax', name='final_predictions')(model.layers[-1].output)
model = Model(inputs=model.input, outputs=output_layer)

# Verifica arhitectura modelului
model.summary()

# Înghețăm greutățile tuturor straturilor
for layer in model.layers[:-1]:
    layer.trainable = False

# Deblocăm greutățile ultimului strat
model.layers[-1].trainable = True


model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

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


batch_size = 256
start_load = datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S")
print("start time:", start_load.strftime("%H:%M:%S"))

dataset_train_val = load_dataset(file_pattern=output_file_train_val, batch_size=batch_size)
dataset_test = load_dataset(file_pattern=output_file_test, batch_size=batch_size)
with open(output_labels_train_val, 'rb') as f:
    y_train_val_all = np.array(pickle.load(f))

with open(output_labels_test, 'rb') as f:
    y_test_all = np.array(pickle.load(f))

#y_train_val_all = LabelEncoder().fit_transform(y_train_val_all)
#y_test_all = LabelEncoder().fit_transform(y_test_all)

end_load = datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S")
print("end time:", end_load.strftime("%H:%M:%S"))
elapsed = end_load - start_load
print(f"elapsed: {elapsed}")

write_msg(f"Incarcarea valorilor spectrogramelor conform fisierelor de pe emodb a durat {elapsed}\n")

antrenari = []
i = 0
start_train = datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S")
print("start time:", start_train.strftime("%H:%M:%S"))

for batch_train_val, batch_test in zip(dataset_train_val.as_numpy_iterator(), dataset_test.as_numpy_iterator()):
    start, end = i * batch_size, (i + 1) * batch_size
    if end > y_train_val_all.shape[0]:
        end = -1
    X_train_val = batch_train_val[:, :, :, :3]
    X_test = batch_test[:, :, :, :3]

    y_train_val = y_train_val_all[start:end]
    y_test = y_test_all[start:end]

    print(f"X_train_val: {X_train_val.shape}, batch no: {i + 1}")
    print(f"X_test: {X_test.shape}, batch no: {i + 1}")
    print(f"y_train_val: {y_train_val.shape}, batch no: {i + 1}")
    print(f"y_test: {y_test.shape}, batch no: {i + 1}")

    write_msg(f"X_train_val: {X_train_val.shape}, batch no: {i + 1}\n")
    write_msg(f"X_test: {X_test.shape}, batch no: {i + 1}\n")
    write_msg(f"y_train_val: {y_train_val.shape}, batch no: {i + 1}\n")
    write_msg(f"y_test: {y_test.shape}, batch no: {i + 1}\n")

    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42) # 70% date de antrenare, 30% date de testare
    # random_state = 42 means that the split is reproducible, meaning that if the script is run again with the same random_state, the data will be split in the same way
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)

    print(f"X_train.shape = {X_train.shape}, batch no: {i + 1}")
    print(f"y_train.shape = {y_train.shape}, batch no: {i + 1}")
    print(f"X_val.shape = {X_val.shape}, batch no: {i + 1}")
    print(f"y_val.shape = {y_val.shape}, batch no: {i + 1}")
    print(f"X_test.shape = {X_test.shape}, batch no: {i + 1}")
    print(f"y_test.shape = {y_test.shape}, batch no: {i + 1}")

    write_msg(f"X_train.shape = {X_train.shape}, batch no: {i + 1}\n")
    write_msg(f"y_train.shape = {y_train.shape}, batch no: {i + 1}\n")
    write_msg(f"X_val.shape = {X_val.shape}, batch no: {i + 1}\n")
    write_msg(f"y_val.shape = {y_val.shape}, batch no: {i + 1}\n")
    write_msg(f"X_test.shape = {X_test.shape}, batch no: {i + 1}\n")
    write_msg(f"y_test.shape = {y_test.shape}, batch no: {i + 1}\n")

    history = model.fit(X_train, y_train, batch_size=15, epochs=15, validation_data=(X_val, y_val))
    antrenari.append(history)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"accuracy: {accuracy}, batch_no: {i + 1}")
    write_msg(f"accuracy: {accuracy}, batch_no: {i + 1}\n")
    i += 1

end_train = datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S")
print("end time:", end_train.strftime("%H:%M:%S"))
elapsed = end_train - start_train
print(f"elapsed: {elapsed}")

write_msg(f"Antrenarea modelului pe baza fisierelor de pe emodb a durat {elapsed}\n")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"accuracy: {accuracy}; loss: {loss}")
write_msg(f"La finalul antrenarii am: accuracy = {accuracy}; loss = {loss}\n")

filename_model = "model_vgg16.h5"
model.save(filename_model)
print(f"Am salvat modelul cu denumirea {filename_model}")
write_msg(f"Am salvat modelul cu denumirea {filename_model}\n")

def plot_training_history(history, filename):
    # Extrage valorile pentru pierdere și acuratețe din istoricul antrenamentului
    loss = history.history['loss']
    write_msg("loss: ")
    for l in loss:
        write_msg(f"{l} ")
    val_loss = history.history['val_loss']
    write_msg("\nval_loss: ")
    for vl in val_loss:
        write_msg(f"{vl} ")
    accuracy = history.history['accuracy']
    write_msg("\ntrain_accuracy: ")
    for a in accuracy:
        write_msg(f"{a} ")
    val_accuracy = history.history['val_accuracy']
    write_msg("\nval_accuracy: ")
    for va in val_accuracy:
        write_msg(f"{va} ")
    
    write_msg("\n")

    # Creați graficul pentru pierdere
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Creați graficul pentru acuratețe
    plt.subplot(1, 2, 2)
    plt.plot(accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Afișați graficele
    plt.tight_layout()
    plt.show()
    plt.savefig(filename)


dir_imagini = "imagini"

if not os.path.exists(dir_imagini):
    os.mkdir(dir_imagini)
    
dir_imagini += "/imagine"
nr_img = 1
ext = ".png"

for antrenare in antrenari:
    filename = dir_imagini + str(nr_img) + ext
    plot_training_history(antrenare, filename)
    nr_img += 1
