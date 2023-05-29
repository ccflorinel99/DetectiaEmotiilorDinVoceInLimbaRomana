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
    if os.path.exists(output_file_train_val):
        os.remove(output_file_train_val)
    if os.path.exists(output_file_test):
        os.remove(output_file_test)
    if os.path.exists(output_labels_train_val):
        os.remove(output_labels_train_val)
    if os.path.exists(output_labels_test):
         os.remove(output_labels_test)
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
                'N' : 5,
                'S' : 6
                }

path = "emodb/wav"

files, output_file_train_val, output_file_test, output_labels_train_val, output_labels_test = get_all(path)

A, B, D, F, H, S, N = impartire_dupa_emotii(path)
A_impartit, B_impartit, D_impartit, F_impartit, H_impartit, S_impartit, N_impartit = impartire_fisiere_pt_toate_emotiile(A, B, D, F, H, S, N)

count_file = 1
s = datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S")
start_time = s.strftime("%H:%M:%S")
print(f"start time: {start_time}")


writer_train_val, writer_test, writer_labels_train_val, writer_labels_test = get_writers(output_file_train_val, output_file_test, output_labels_train_val, output_labels_test)
labels_train_val, labels_test = [], []


for file in files:
    file_start_time = datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S")

    audio, sr = librosa.load(path + "/" + file)

    # Calculați lungimea fiecărui cadru în eșantioane
    frame_length = int(sr)
    hop_length = int(sr * 0.01)

    # Segmentați fișierul audio în cadre de 1 secundă cu un pas de 10 ms
    audio_frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    #print(audio_frames.shape)
    i = 0
    #hopLengthinSpectrogram=int(sr*0.03)

    for frame in audio_frames.T:
        #print(frame.shape)
        start, end = 0 + i * 0.01, 1 + i * 0.01
        log = spectrograma(frame, sr, start=start, end=end, show=False, n_fft=2048, n_mels=64, hop_length=int(sr*0.01))
        log_tf_record = convert_to_tfrecord(log)

        if e_in_liste(file, A_impartit, B_impartit, D_impartit, F_impartit, H_impartit, S_impartit, N_impartit):
            writer_test.write(log_tf_record)
            labels_test.append(codif_emotii[get_emotion(file)])
        else:
            writer_train_val.write(log_tf_record)
            labels_train_val.append(codif_emotii[get_emotion(file)])
        i += 1

    file_end_time = datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S")
    print(f"{count_file}/{len(files)} done ({file_end_time - file_start_time})")
    count_file += 1

pickle.dump(labels_train_val, writer_labels_train_val)
pickle.dump(labels_test, writer_labels_test)
writer_train_val.close()
writer_test.close()
writer_labels_train_val.close()
writer_labels_test.close()
e = datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S")
current_time = e.strftime("%H:%M:%S")
print(f"end time: {current_time}")
print(f"elapsed: {e - s}")

write_msg("detalii_antrenare.txt", f"Preprocesarea fisierelor de pe emodb a durat {e - s}\n")
