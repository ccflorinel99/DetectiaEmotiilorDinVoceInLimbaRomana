import os
from datetime import datetime
import tensorflow as tf
import librosa
import numpy as np
import matplotlib.pyplot as plt
import cv2


def spectrograma(audio, sr, n_fft=2048, hop_length=160, n_mels=40, start=0, end=1, show=True):
    # Calculati spectograma si transformati-o în decibeli
    spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_spec = librosa.amplitude_to_db(spec, ref=np.max)
    #log_spec = spec
    # Aplicati o transformare liniara pentru a obtine valorile in intervalul 0-255
    v_min = -90
    v_max = -7
    log_spec = (log_spec - v_min) / (v_max - v_min) * 255
    log_spec = np.clip(log_spec, 0, 255)

    # Transformati spectograma logaritmica în RGB
    cmap = plt.get_cmap('jet')
    log_spec_rgb = cmap(log_spec.astype(np.uint8))

    if show:
        # Creeaza o bara de culoare personalizata cu culori ordonate
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
    
    # Redimensionati spectrograma si transformati-o într-un tensor de 3 dimensiuni
    log_spec_rgb = cv2.resize(log_spec_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    log_spec_rgb_tensor = tf.keras.preprocessing.image.img_to_array(log_spec_rgb)

    # Normaliza?i tensorul la intervalul [0, 1]
    #log_spec_rgb_tensor /= 255.0

    return log_spec_rgb_tensor

def convert_to_tfrecord(value):
    value = value.astype(np.float32)  # Asigura-te ca valorile sunt de tipul np.float32
    feature = {
        'value': tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten())),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def progres(folder_spectrograme):
    spectrograme = len(os.listdir(folder_spectrograme))
    p = round(spectrograme * 100 / 42, 2)
    return f"Progres: {p}%"


path = "dataset"
director_spectrograme = "spectrograme"
if not os.path.exists(director_spectrograme):
    os.mkdir(director_spectrograme)

files = os.listdir(path)
counter = 1

for file in files:
    outfile = file[:-3] + "bin"
    writer = tf.io.TFRecordWriter(director_spectrograme + "/" + outfile)
    audio, sr = librosa.load(path + "/" + file)

    # Calcula?i lungimea fiecarui cadru în e?antioane
    frame_length = int(sr)
    hop_length = int(sr * 0.01)

    # Segmenta?i fi?ierul audio în cadre de 1 secunda cu un pas de 10 ms
    audio_frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    #print(audio_frames.shape)
    i = 0
    print(f"file: {file}, file_no: {counter}, frames: {len(audio_frames.T)} is starting")
    #hopLengthinSpectrogram=int(sr*0.03)
    file_start_time = datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S")
    for frame in audio_frames.T:
        #print(frame.shape)
        start, end = 0 + i * 0.01, 1 + i * 0.01
        log = spectrograma(frame, sr, start=start, end=end, show=False, n_fft=2048, n_mels=64, hop_length=int(sr*0.01))
        log_tf_record = convert_to_tfrecord(log)
        writer.write(log_tf_record)
        i += 1
        if i == round(len(audio_frames.T) / 2):
            current_time = datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S")
            print(f"Halfway there. Elapsed untill now {current_time - file_start_time}")
        
    file_end_time = datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S")
    print(f"{outfile} done ({file_end_time - file_start_time})")
    writer.close()
    print(f"{counter}/{len(files)} done, {progres('spectrograme')}")
    counter += 1

