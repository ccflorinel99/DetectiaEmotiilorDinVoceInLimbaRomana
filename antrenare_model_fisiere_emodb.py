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

#cmap='jet'

def spectrograma(audio, sr, n_fft=2048, hop_length=160, n_mels=40, start=0, end=1, show=True):
    # Calculați spectograma și transformați-o în decibeli
    spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_spec = librosa.amplitude_to_db(spec, ref=np.max)
    #log_spec = spec
    # Aplicați o transformare liniară pentru a obține valorile în intervalul 0-255
    v_min = -90
    v_max = -7
    log_spec = (log_spec - v_min) / (v_max - v_min) * 255
    log_spec = np.clip(log_spec, 0, 255)

    # Transformați spectograma logaritmică în RGB
    cmap = plt.get_cmap('jet')
    log_spec_rgb = cmap(log_spec.astype(np.uint8))

    if show:
        # Creează o bară de culoare personalizată cu culori ordonate
        mycolors = [(0.8,0,0), (1,0,0), (1,0.8,0), (1,1,0), (0,1,1), (0,0.2,0.8), (0,0,1), (0,0,0.4)]
        mycolors.reverse()
        boundaries = [-80, -70, -60, -50, -40, -30, -20, -10, 0]
        # Creeati o harta de culori personalizata cu segmentari personalizate
        cmap_seg = ListedColormap(mycolors)
        norm = BoundaryNorm(boundaries, cmap_seg.N, clip=True)

        # Adaugati durata segmentului audio in secunde
        duration = len(audio) / sr
        t = np.arange(0, duration, duration / log_spec.shape[1])

        # Afisati spectrograma cu noua harta de culori si segmentari personalizate
        fig, ax = plt.subplots()
        im = ax.imshow(log_spec_rgb, aspect='auto', origin='lower', cmap=cmap_seg, norm=norm, extent=[0, duration, 0, sr/2])


        # Adaugati colorbar personalizat cu segmentari si culori
        cbar = fig.colorbar(im, ax=ax, ticks=boundaries)
        cbar.ax.set_yticklabels([f'{i}' for i in boundaries])
        ax.set_xlabel(f"Timp (s) (durata {duration}, din original: {start}-{end})")
        ax.set_ylabel("Hz")
        plt.show()
    
    # Redimensionați spectrograma și transformați-o într-un tensor de 3 dimensiuni
    log_spec_rgb = cv2.resize(log_spec_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    log_spec_rgb_tensor = tf.keras.preprocessing.image.img_to_array(log_spec_rgb)

    # Normalizați tensorul la intervalul [0, 1]
    #log_spec_rgb_tensor /= 255.0

    return log_spec_rgb_tensor

def get_emotion(file):
    em = file[5:(-1)*len(".wav")]
    if em[0] == 'W':
        em = 'A'
    elif em[0] == 'L':
        em = 'B'
    elif em[0] == 'E':
        em = 'D'
    elif em[0] == 'A':
        em = 'F'
    elif em[0] == 'F':
        em = 'H'
    elif em[0] == 'T':
        em = 'S'
    elif em[0] == 'N':
        em = 'N'
        
    return em

def convert_to_tfrecord(value):
    value = value.astype(np.float32)  # Asigură-te că valorile sunt de tipul np.float32
    feature = {
        'value': tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten())),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def get_all(path):
    files, output_file_train_val, output_file_test, output_labels_train_val, output_labels_test = os.listdir(path), 'spectrograme_train_val.bin', 'spectrograme_test.bin', 'labels_train_val.bin', 'labels_test.bin'
    
    return files, output_file_train_val, output_file_test, output_labels_train_val, output_labels_test

def impartire_dupa_emotii(folder):
    A = []
    B = []
    D = []
    F = []
    H = []
    S = []
    N = []

    files = os.listdir(folder)
    for file in files:
        em = get_emotion(file)
        if em == 'A':
            A.append(file)
        elif em == 'B':
            B.append(file)
        elif em == 'D':
            D.append(file)
        elif em == 'F':
            F.append(file)
        elif em == 'H':
            H.append(file)
        elif em == 'S':
            S.append(file)
        elif em == 'N':
            N.append(file)
            
    return A, B, D, F, H, S, N


def get_procent(lista, procent=0.1):
    num_samples = int(len(lista) * procent)
    random_sample = random.sample(lista, num_samples)
    return random_sample

def impartire_fisiere_pt_toate_emotiile(A, B, D, F, H, S, N, procent=0.1):
    A_impartit = get_procent(A, procent=procent)
    B_impartit = get_procent(B, procent=procent)
    D_impartit = get_procent(D, procent=procent)
    F_impartit = get_procent(F, procent=procent)
    H_impartit = get_procent(H, procent=procent)
    S_impartit = get_procent(S, procent=procent)
    N_impartit = get_procent(N, procent=procent)
    return A_impartit, B_impartit, D_impartit, F_impartit, H_impartit, S_impartit, N_impartit

def e_in_lista(file, lista):
    gasit = False
    for el in lista:
        if file == el:
            gasit = True
    return gasit

def e_in_liste(file, A_impartit, B_impartit, D_impartit, F_impartit, H_impartit, S_impartit, N_impartit):
    return e_in_lista(file, A_impartit) or e_in_lista(file, B_impartit) or e_in_lista(file, D_impartit) or e_in_lista(file, F_impartit) or e_in_lista(file, H_impartit) or e_in_lista(file, S_impartit) or e_in_lista(file, N_impartit)


def get_writers(output_file_train_val, output_file_test, output_labels_train_val, output_labels_test):
    writer_train_val = tf.io.TFRecordWriter(output_file_train_val)
    writer_test = tf.io.TFRecordWriter(output_file_test)
    writer_labels_train_val = open(output_labels_train_val, 'wb')
    writer_labels_test = open(output_labels_test, 'wb')
    return writer_train_val, writer_test, writer_labels_train_val, writer_labels_test

def write_msg(output_file, msg):
    f = open(output_file, "a", encoding="utf-8")
    f.write(msg)
    f.close()
    
    
codif_emotii = {
                'A' : 0,
                'B' : 1,
                'D' : 2,
                'F' : 3,
                'H' : 4,
                'I' : 5,
                'N' : 6,
                'S' : 7
                }

num_classes = len(codif_emotii)

path = "emodb/wav"

files, output_file_train_val, output_file_test, output_labels_train_val, output_labels_test = get_all(path)

# Incarca modelul VGG16 preantrenat
model = VGG16(weights='imagenet', include_top=True)

# Inlatura layer-ul de output original
model.layers.pop()

# Adauga un nou output layer cu 8 clase
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
    y_train_val_all = pickle.load(f)

with open(output_labels_test, 'rb') as f:
    y_test_all = pickle.load(f)

y_train_val_all = LabelEncoder().fit_transform(y_train_val_all)
y_test_all = LabelEncoder().fit_transform(y_test_all)

end_load = datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S")
print("end time:", end_load.strftime("%H:%M:%S"))
elapsed = end_load - start_load
print(f"elapsed: {elapsed}")

write_msg("detalii_antrenare.txt", f"Incarcarea valorilor spectrogramelor conform fisierelor de pe emodb a durat {elapsed}\n")

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

    write_msg("detalii_antrenare.txt", f"X_train_val: {X_train_val.shape}, batch no: {i + 1}\n")
    write_msg("detalii_antrenare.txt", f"X_test: {X_test.shape}, batch no: {i + 1}\n")
    write_msg("detalii_antrenare.txt", f"y_train_val: {y_train_val.shape}, batch no: {i + 1}\n")
    write_msg("detalii_antrenare.txt", f"y_test: {y_test.shape}, batch no: {i + 1}\n")

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

    write_msg("detalii_antrenare.txt", f"X_train.shape = {X_train.shape}, batch no: {i + 1}\n")
    write_msg("detalii_antrenare.txt", f"y_train.shape = {y_train.shape}, batch no: {i + 1}\n")
    write_msg("detalii_antrenare.txt", f"X_val.shape = {X_val.shape}, batch no: {i + 1}\n")
    write_msg("detalii_antrenare.txt", f"y_val.shape = {y_val.shape}, batch no: {i + 1}\n")
    write_msg("detalii_antrenare.txt", f"X_test.shape = {X_test.shape}, batch no: {i + 1}\n")
    write_msg("detalii_antrenare.txt", f"y_test.shape = {y_test.shape}, batch no: {i + 1}\n")

    history = model.fit(X_train, y_train, batch_size=15, epochs=15, validation_data=(X_val, y_val))
    antrenari.append(history)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"accuracy: {accuracy}, batch_no: {i + 1}")
    write_msg("detalii_antrenare.txt", f"accuracy: {accuracy}, batch_no: {i + 1}\n")
    i += 1

end_train = datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S")
print("end time:", end_train.strftime("%H:%M:%S"))
elapsed = end_train - start_train
print(f"elapsed: {elapsed}")

write_msg("detalii_antrenare.txt", f"Antrenarea modelului pe baza fisierelor de pe emodb a durat {elapsed}\n")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"accuracy: {accuracy}; loss: {loss}")
write_msg("detalii_antrenare.txt", f"La finalul antrenarii am: accuracy = {accuracy}; loss = {loss}\n")

filename_model = "model_vgg16.h5"
model.save(filename_model)
print(f"Am salvat modelul cu denumirea {filename_model}")
write_msg("detalii_antrenare.txt", f"Am salvat modelul cu denumirea {filename_model}")

def plot_training_history(history, filename):
    # Extrage valorile pentru pierdere și acuratețe din istoricul antrenamentului
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

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
