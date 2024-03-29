# https://data-flair.training/blogs/python-mini-project-speech-emotion-recognition/ # model de cod
# pentru implementarea codului am urmat ideile din urmatoarele link-uri:
# 1. https://github.com/topics/speech-emotion-recognition
# 2. https://github.com/MiteshPuthran/Speech-Emotion-Analyzer
# 3. https://github.com/xuanjihe/speech-emotion-recognition
# 4. https://github.com/Demfier/multimodal-speech-emotion-recognition
# 5. https://github.com/hkveeranki/speech-emotion-recognition
# 6. https://github.com/x4nth055/emotion-recognition-using-speech
# https://www.researchgate.net/figure/The-SED-System-a-Real-Time-Speech-Emotion-Detection-System-based-on-Internet-of-Things_fig1_337992475 # discutii

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# type of generalized linear model (GLM) that uses a logistic function to model a binary dependent variable. It estimates the probability that a given input belongs to a particular class and then classifies it based on a threshold (supervised)
from sklearn.tree import DecisionTreeClassifier
# flowchart-like structure, where each internal node represents a feature(or attribute), each branch represents a decision rule, and each leaf node represents the outcome (supervised)
from sklearn.neural_network import MLPClassifier
# type of feedforward artificial neural network that consists of one or more layers of artificial neurons, called perceptrons. Each perceptron receives input from the previous layer and applies a non-linear activation function to it before passing the output to the next layer. The last layer of perceptrons is called the output layer and it generates the final predictions. The algorithm learns the weights of the perceptrons by minimizing the difference between the predicted output and the true output using an optimization algorithm, such as stochastic gradient descent. (supervised)
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier # multiple trees and combines them to make a decision by using "vote system" and the majority wins (supervised)
import os
import pickle
from sklearn.svm import SVC # find the best boundary (supervised)
from pydub import AudioSegment
from IPython.display import Audio, display
import soundfile as sf
import sounddevice as sd
import sys
from scipy.io.wavfile import write
from tkinter import *
from tkinter import filedialog
import keyboard
import time
import librosa
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
import cv2
from datetime import datetime
from tabulate import tabulate



class Files:
    def __init__(self):
        pass
    
    def path_exists(self, path):
        return os.path.exists(path)

class Output():
    def __init__(self):
        self.msg = ""

    def add(self, msg):
        if self.msg == "":
            self.msg = msg
        else:
            self.msg = self.msg + "\n" + msg

    def show(self):
        print(self.msg)

    def clear(self):
        self.msg = ""

    def have_output(self):
        if self.msg == "":
            return False
        else:
            return True


class Error():
    def __init__(self):
        self.msg = ""
        self.out = Output()

    def add_error(self, msg):
        if self.msg == "":
            self.msg = msg
        else:
            self.msg = self.msg + "\n" + msg
  
    def path_not_found(self, path):
        self.add_error(f"{path} nu există")

    def got_error(self):
        if self.msg == "":
            return False
        else:
            return True

    def clear(self):
        self.msg = ""

    def show(self):
        print(self.msg)


out = Output()
err = Error()

class Emotions:
    def __init__(self): # codificare emotii
        self.d = {
                    0 : "A",
                    1 : "B",
                    2 : "D",
                    3 : "F",
                    4 : "H",
                    5 : "I",
                    6 : "N",
                    7 : "S"
                }
        
    def to_list(self):
        lista = []
        for key in self.d:
            lista.append(self.d[key])
        return lista
    
    def get_emotion(self, code : int): # code e variabila unde se va stoca ce emotie a fost prezisa (0-7)
        found = True
        for key in self.d:
            if code == key:
                return self.d[key]
            
        if not found:
            err.add_error(f"Emoția prezisă nu se află în dicționar, aceasta având codul {code}")
            
            
class AudioManipulation:
    def __init__(self):
        self.files = Files()

    def load(self, file_path):
        if self.files.path_exists(file_path):
            audio = AudioSegment.from_wav(file_path)
            # sr = sample rate
            sr = audio.frame_rate  
            return audio, sr
# Sample rate, sau "rata de eșantionare", reprezintă numărul de eșantioane de semnal luate pe secundă.
# Este exprimat în herți (Hz) și reprezintă numărul de câți eșantioane sunt luate pe secundă din semnalul audio.
# Rata de eșantionare afectează calitatea sunetului și frecvența maximă care poate fi înregistrată sau redată.
# Cu cât rata de eșantionare este mai mare, cu atât semnalul audio este redat mai precis.
# O rata de eșantionare mai mare va permite redarea frecvențelor mai înalte, dar va ocupa mai mult spațiu pe disc.
# Rata standard pentru aplicațiile audio digitale este de 44.1 sau 48 kHz.
        else:
            err.path_not_found(file_path)
            
class Preprocessing():
    def __init__(self):
        self.files = Files()
        self.am = AudioManipulation()
        self.emodb = "C:/Users/User/Desktop/Licenta/emodb/wav"
        
    def get_emotion_emodb(self, file):
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
        
    def pereche_wav_txt(self, dir : str):
        self.dir = dir
        if not self.files.path_exists(dir): # verifica daca directorul exista sau nu
            err.path_not_found(dir)
        else:
      # listam fisierele din directorul dat ca parametru
            filenames = os.listdir(self.dir)

            if self.emodb == self.dir: # daca fac antrenare pe fisierele de pe emodb
                self.audio_emotion_pair = []
                for filename in filenames:
                    em = self.get_emotion_emodb(filename)
                    full_path = os.path.join(self.dir, filename)
                    lista = [full_path, em]
                    self.audio_emotion_pair.append(lista)
            else:
                # cream lista de tupluri formata din numele fisierelor audio si numele fisierelor txt asociate
                self.audio_txt_pairs = []
                for filename in filenames:
                    if filename.endswith('.wav'):
                        txt_filename = filename.replace('.wav', '.txt')
                        full_path = os.path.join(self.dir, txt_filename)
                        if self.files.path_exists(full_path):
                            self.audio_txt_pairs.append((os.path.join(self.dir, filename), os.path.join(self.dir, txt_filename)))
                        else:
                            err.path_not_found(full_path)

    def preprocess_audio_file(self, audio_file, txt_file):
        if err.got_error():
            out.add("Ai de rezolvat una sau mai multe erori înainte de a lucra cu funcția preprocess_audio_file")
            return [0], 0
        else: 
      # incarcam fisierul audio
            audio, sr = self.am.load(audio_file)
            
            if self.emodb == self.dir: # daca fac antrenare pe fisierele de pe emodb
                return audio, sr
            else:
                audios = []
    
                if(self.files.path_exists(txt_file)):
                    with open(txt_file, 'r', encoding="utf-8") as f:
                        lines = f.readlines()
                        # impartim fisierul audio in fragmente
                        for line in lines:
                            start = float(line[:line.find("[")].split(' ')[0].split('\t')[0]) * 1000
                            end = float(line[:line.find("[")].split(' ')[0].split('\t')[1]) * 1000
                            fragment_audio = audio[start:end]
                            # adaugam fragmentul audio la lista de fragmente
                            audios.append(fragment_audio)
                        f.close()
                else:
                    err.path_not_found(txt_file)
    
            return audios, sr

    def extract_features(self, audio, sr):
        if err.got_error():
            out.add("Ai de rezolvat una sau mai multe erori înainte de a lucra cu funcția extract_features")
            return [0, 0, 0, 0, 0, 0]
        else: 
            # transformare din pydub audiosegment in numpy array
            audio = np.asarray(audio.get_array_of_samples(),dtype = np.float64)
            # calculam amplitudinea semnalului
            amplitudes = np.abs(audio)

            # calculam spectrul de frecventa
            spectrogram = np.abs(librosa.stft(audio))

            # calculam rata de variatie a frecventei vorbirii
            pitch = librosa.piptrack(y=audio, sr=sr)
            pitch = np.array([pitch])
            pitch_change = np.diff(pitch)
            pitch_change_rate = 0
            if len(pitch_change) != 0:
                pitch_change_rate = np.mean(pitch_change)

            # calculam lungimea silabelor
            syllable_lengths = []
            for i in range(len(audio) - 1):
                if audio[i] > 0 and audio[i + 1] < 0:
                    syllable_lengths.append(i + 1 - sum(syllable_lengths))

            # calculam durata pauzelor
            pause_lengths = []
            for i in range(len(audio) - 1):
                if audio[i] == 0 and audio[i + 1] != 0:
                    pause_lengths.append(i + 1 - sum(pause_lengths))

            # calculam rata de respiratie
            respiration_rate = 0
            if(sum(syllable_lengths) != 0):
                respiration_rate = len(syllable_lengths) / sum(syllable_lengths)

            # returnam caracteristicile
            a = np.array([np.mean(amplitudes), np.mean(spectrogram), pitch_change_rate, np.mean(syllable_lengths), np.mean(pause_lengths), respiration_rate])
            # pitch_change_rate si respiration_rate au cele mai mari sanse sa dea nan  
            a = np.nan_to_num(a)
            
            return np.mean(a)#np.array(np.mean(a))

  
    def preprocesare(self):
        if err.got_error():
            out.add("Ai de rezolvat una sau mai multe erori înainte de a lucra cu funcția preprocesare")
            return 0, 0
        else: 
            X = []
            labels = []
            if self.emodb == self.dir: # daca fac antrenare pe fisierele de pe emodb
                audio_files = [pair[0] for pair in self.audio_emotion_pair]
                em = [pair[1] for pair in self.audio_emotion_pair]
                for i in range(len(audio_files)):
                    audio, sr = self.preprocess_audio_file(audio_files[i], em[i])
                    labels.append(em[i])
                    features = self.extract_features(audio, sr)
                    X = np.append(X, features)
            else:
                # obtinem listele audio_files si txt_files
                audio_files = [pair[0] for pair in self.audio_txt_pairs]
                txt_files = [pair[1] for pair in self.audio_txt_pairs]

                # citim etichetele de emotie din fisierele txt
                for txt_file in txt_files:
                    with open(txt_file, 'r', encoding="utf-8") as f:
                        lines = f.readlines()
                        for line in lines:
                            label = line[line.find("["):].split(' ')[0].replace("[", "").replace("]", "")
                            labels.append(label)

                # extragem caracteristicile din fisierele audio

                for i in range(len(self.audio_txt_pairs)):
                    audio, sr = self.preprocess_audio_file(audio_files[i], txt_files[i])
                    for j in range(len(audio)):
                        features = self.extract_features(audio[j], sr)
                        X = np.append(X, features)

            X = X.reshape(-1, 1)
            
    
            # transformam etichetele de emotie in forma utilizabila de model
            le = LabelEncoder()
            y = le.fit_transform(labels)

            return X, y
        
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
    
    def get_N_codif(self):
        return self.dict["N"]
    
    def decode_list(self, encoded_list):
        dec_list = []
        for elem in encoded_list:
            for key in self.dict:
                if elem == self.dict[key]:
                    dec_list.append(key)
                    
        return dec_list
    
    def decode_value(self, value):
        for key in self.dict:
            if self.dict[key] == value:
                return key
    

class VGG16_Modif:
    def __init__(self):
        self.model_emodb = keras.models.load_model("model_vgg16.h5")
        self.model_dataset = keras.models.load_model("model_vgg16_dataset.h5")
        self.dict = {"emodb": 0,
                    "dataset": 0}
        f = open("models_accuracy.txt", "r")
        lines = f.readlines()
        f.close()

        for line in lines:
            l = line.split(":")
            acc = float(l[1].split("\n")[0])
            self.dict[l[0]] = acc
        
    def spectrograma(self, audio, sr, n_fft=2048, hop_length=160, n_mels=64, start=0, end=1, show=True):
        # Calculati spectograma si transformati-o in decibeli
        spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        log_spec = librosa.amplitude_to_db(spec, ref=np.max)
        #log_spec = spec
        # Aplicati o transformare liniara pentru a obtine valorile in intervalul 0-255
        v_min = -90
        v_max = -7
        log_spec = (log_spec - v_min) / (v_max - v_min) * 255
        log_spec = np.clip(log_spec, 0, 255)

        # Transformati spectograma logaritmica in RGB
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
    
        # Redimensionati spectrograma si transformati-o intr-un tensor de 3 dimensiuni
        log_spec_rgb = cv2.resize(log_spec_rgb, (224, 224), interpolation=cv2.INTER_AREA)
        log_spec_rgb_tensor = tf.keras.preprocessing.image.img_to_array(log_spec_rgb)

        # Normalizati tensorul la intervalul [0, 1]
        #log_spec_rgb_tensor /= 255.0

        return log_spec_rgb_tensor
        
    def use(self, filename, mle):
        outputs = []
        audio, sr = librosa.load(filename)

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
            log = self.spectrograma(frame, sr, start=start, end=end, show=False, hop_length=hop_length)
            log = np.array([log[:, :, :3]])
            outputs.append(self.model.predict(log))
            print(f"{i+1}/{len(audio_frames.T)} done")
            i += 1
    
        lista = []

        for o in outputs:
            for elem in o[0]:
                lista.append(elem)

        return mle.decode_value(round(np.mean(np.array(lista))))

    def get_dict(self, lista):
        #print(np.array(lista).all() == np.array(lista_load).all())
        a = np.array(lista)
        # Numărați valorile unice și frecvența lor
        unique_values, counts = np.unique(a, return_counts=True)

        # Crearea unui dicționar pentru a cataloga valorile și frecvența lor
        value_counts = {}

        # Iterați prin valorile unice și frecvența lor și adăugați-le în dicționar
        for value, count in zip(unique_values, counts):
            if np.isnan(value):
                value = 'nan'
            value_counts[value] = count
            
        return value_counts
    
    def get_dict_fara_nan(self, lista):
        #print(np.array(lista).all() == np.array(lista_load).all())
        a = np.array(lista)
        # Numărați valorile unice și frecvența lor
        unique_values, counts = np.unique(a, return_counts=True)

        # Crearea unui dicționar pentru a cataloga valorile și frecvența lor
        value_counts = {}

        # Iterați prin valorile unice și frecvența lor și adăugați-le în dicționar
        for value, count in zip(unique_values, counts):
            value_counts[value] = count
            
        return value_counts
    
    def vot(self, array_emotii):
        maxim = 0

        for key in array_emotii:
            if key != "nan":
                if maxim < array_emotii[key]:
                    maxim = array_emotii[key]
            
        msg = ""
        if maxim == 0: # toate valorile sunt nan
            msg = "N" # bag neutru ca fiind emotia
        else: # am si alte valori
            for key in array_emotii:
                if maxim == array_emotii[key]:
                    if msg == "":
                        msg = key
                    else:
                        msg += f"/{key}"

        return msg
    
    def get_accuracy(self, for_what):
        return self.dict[for_what]
        

class Models:
    def __init__(self):
        self.am = AudioManipulation()
        self.emotions = Emotions()
        # creeam modelele
        self.RandomForestClassifierModel = RandomForestClassifier()
        self.SVCModel = SVC()
        self.LogisticRegressionModel = LogisticRegression(max_iter=5000)
# daca lasam default (max_iter=100 default) sau max_iter=500, atunci imi dadea eroarea STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
# discutie: https://stackoverflow.com/questions/62658215/convergencewarning-lbfgs-failed-to-converge-status-1-stop-total-no-of-iter
        self.DecisionTreeClassifierModel = DecisionTreeClassifier()
        self.MLPClassifierModel = MLPClassifier()
        self.files = Files()

    def save_models(self):
        if err.got_error():
            out.add("Ai de rezolvat una sau mai multe erori înainte de a salva modelele")
        else:
            if not self.files.path_exists("models"):
                os.makedirs("models")
                
            with open("models/RandomForestClassifierModel.pkl", "wb") as f:
                # salvam modelul in fisier
                pickle.dump(self.RandomForestClassifierModel, f)
            out.add("models/RandomForestClassifierModel a fost salvat")

            with open("models/SVCModel.pkl", "wb") as f:
                # salvam modelul in fisier
                pickle.dump(self.SVCModel, f)
            out.add("models/SVCModel a fost salvat")

            with open("models/LogisticRegressionModel.pkl", "wb") as f:
                # salvam modelul in fisier
                pickle.dump(self.LogisticRegressionModel, f)
            out.add("models/LogisticRegressionModel a fost salvat")

            with open("models/DecisionTreeClassifierModel.pkl", "wb") as f:
                # salvam modelul in fisier
                pickle.dump(self.DecisionTreeClassifierModel, f)
            out.add("models/DecisionTreeClassifierModel a fost salvat")

            with open("models/MLPClassifierModel.pkl", "wb") as f:
                # salvam modelul in fisier
                pickle.dump(self.MLPClassifierModel, f)
            out.add("models/MLPClassifierModel a fost salvat")

    def load_models(self):
        if not self.files.path_exists("models"):
            err.add_error("Folderul 'models' nu există, rezultă faptul că va trebui să antrenezi întâi modelele")
        else:
            count = 0
            if not self.files.path_exists("models/RandomForestClassifierModel.pkl"):
                err.add_error("Fișierul 'models/RandomForestClassifierModel.pkl' nu există")
                count = count + 1
            if not self.files.path_exists("models/SVCModel.pkl"):
                err.add_error("Fișierul 'models/SVCModel.pkl' nu exista")
                count = count + 1
            if not self.files.path_exists("models/LogisticRegressionModel.pkl"):
                err.add_error("Fișierul 'models/LogisticRegressionModel.pkl' nu există")
                count = count + 1
            if not self.files.path_exists("models/DecisionTreeClassifierModel.pkl"):
                err.add_error("Fișierul 'models/DecisionTreeClassifierModel.pkl' nu există")
                count = count + 1
            if not self.files.path_exists("models/MLPClassifierModel.pkl"):
                err.add_error("Fișierul 'models/MLPClassifierModel.pkl' nu există")
                count = count + 1
                
            if count == 5: # daca nu exista niciun model
                err.add_error("Nu am găsit niciun model, așa că va trebui să antrenezi modelele")
                
        if not err.got_error():
            with open("models/RandomForestClassifierModel.pkl", "rb") as f:
                self.RandomForestClassifierModel = pickle.load(f)
            out.add("models/RandomForestClassifierModel a fost încărcat")

            with open("models/SVCModel.pkl", "rb") as f:
                self.SVCModel = pickle.load(f)
            out.add("models/SVCModel a fost încărcat")

            with open("models/LogisticRegressionModel.pkl", "rb") as f:
                self.LogisticRegressionModel = pickle.load(f)
            out.add("models/LogisticRegressionModel a fost încărcat")

            with open("models/DecisionTreeClassifierModel.pkl", "rb") as f:
                self.DecisionTreeClassifierModel = pickle.load(f)
            out.add("models/DecisionTreeClassifierModel a fost încărcat")

            with open("models/MLPClassifierModel.pkl", "rb") as f:
                self.MLPClassifierModel = pickle.load(f)
            out.add("models/MLPClassifierModel a fost încărcat")

    def train(self, x, y):
        if err.got_error():
            out.add("Nu poți antrena modelele dacă ai erori")
        elif x.size == 0 and y.size == 0:
            err.add_error("Nu s-a realizat preprocesarea")
        else:
            out.add(f"X: {x.shape}")
            out.add(f"y: {y.shape}")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, train_size=0.7, random_state=42) # 70% date de antrenare, 30% date de testare
# random_state = 42 means that the split is reproducible, meaning that if the script is run again with the same random_state, the data will be split in the same way
            out.add(f"X_train: {self.X_train.shape}")
            out.add(f"y_train: {self.y_train.shape}")
            out.add(f"X_test: {self.X_test.shape}")
            out.add(f"y_test: {self.y_test.shape}")
            self.RandomForestClassifierModel.fit(self.X_train, self.y_train)
            self.SVCModel.fit(self.X_train, self.y_train)
            self.LogisticRegressionModel.fit(self.X_train, self.y_train)
            self.DecisionTreeClassifierModel.fit(self.X_train, self.y_train)
            self.MLPClassifierModel.fit(self.X_train, self.y_train)
            out.add("Modelele au fost antrenate")

    def accuracy_score(self, y_true, y_pred):
        num_correct = 0
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                num_correct += 1
        return num_correct / len(y_true)

    def models_accuracy(self):
        if err.got_error():
            out.add("Ai una sau mai multe erori de rezolvat")
        else:
            # evaluam acuratetea modelului pe datele de testare
            self.y_pred_RandomForestClassifier = self.RandomForestClassifierModel.predict(self.X_test)
            self.y_pred_SVC = self.SVCModel.predict(self.X_test)
            self.y_pred_LogisticRegression = self.LogisticRegressionModel.predict(self.X_test)
            self.y_pred_DecisionTreeClassifier = self.DecisionTreeClassifierModel.predict(self.X_test)
            self.y_pred_MLPClassifier = self.MLPClassifierModel.predict(self.X_test)

            self.accuracy_RandomForestClassifier = self.accuracy_score(self.y_test, self.y_pred_RandomForestClassifier)
            self.accuracy_SVC = self.accuracy_score(self.y_test, self.y_pred_SVC)
            self.accuracy_LogisticRegression = self.accuracy_score(self.y_test, self.y_pred_LogisticRegression)
            self.accuracy_DecisionTreeClassifier = self.accuracy_score(self.y_test, self.y_pred_DecisionTreeClassifier)
            self.accuracy_MLPClassifier = self.accuracy_score(self.y_test, self.y_pred_MLPClassifier)


    def create_table(self, accuracy_vgg16):
        #create data
        data = [["RandomForestClassifier Model", self.accuracy_RandomForestClassifier], 
                ["SVC Model", self.accuracy_SVC], 
                ["LogisticRegression Model", self.accuracy_LogisticRegression], 
                ["DecisionTreeClassifier Model", self.accuracy_DecisionTreeClassifier],
                ["MLPClassifier Model", self.accuracy_MLPClassifier],
                ["VGG16 modified Model", accuracy_vgg16]]
  
        #define header names
        col_names = ["Models", "Accuracy"]
  
        #display table
        out.add("\n")
        out.add(tabulate(data, headers=col_names))


    def use_models(self, audio_filename, p : Preprocessing):
        self.predictii = [] # aici voi stoca predictiile
        audio, sr = self.am.load(audio_filename)

        # extrageti caracteristicile fisierului audio
        X = np.array(p.extract_features(audio, sr)).reshape(-1, 1)
        
        # folositi modelele pentru a face predictia
        self.prediction_RandomForestClassifierModel = self.RandomForestClassifierModel.predict(X)
        self.prediction_SVCModel = self.SVCModel.predict(X)
        self.prediction_LogisticRegressionModel = self.LogisticRegressionModel.predict(X)
        self.prediction_DecisionTreeClassifierModel = self.DecisionTreeClassifierModel.predict(X)
        self.prediction_MLPClassifierModel = self.MLPClassifierModel.predict(X)

        # transformati rezultatul predictiei in eticheta de emotie
        label_encoder = LabelEncoder()
        emotion_labels = label_encoder.fit_transform(self.emotions.to_list())
        #out.add(f"Codificări: {emotion_labels}")
        #out.add(f"Emoții: {self.emotions.to_list()}")
        self.emotion_RandomForestClassifierModel = self.emotions.get_emotion(emotion_labels[self.prediction_RandomForestClassifierModel[0]])
        self.emotion_SVCModel = self.emotions.get_emotion(emotion_labels[self.prediction_SVCModel[0]])
        self.emotion_LogisticRegressionModel = self.emotions.get_emotion(emotion_labels[self.prediction_LogisticRegressionModel[0]])
        self.emotion_DecisionTreeClassifierModel = self.emotions.get_emotion(emotion_labels[self.prediction_DecisionTreeClassifierModel[0]])
        self.emotion_MLPClassifierModel = self.emotions.get_emotion(emotion_labels[self.prediction_MLPClassifierModel[0]])
        self.predictii.append(self.emotion_RandomForestClassifierModel)
        self.predictii.append(self.emotion_SVCModel)
        self.predictii.append(self.emotion_LogisticRegressionModel)
        self.predictii.append(self.emotion_DecisionTreeClassifierModel)
        self.predictii.append(self.emotion_MLPClassifierModel)
        
        em = self.predictie()
        
        return em
        
        
    def predictie(self):
        predictii = np.array(self.predictii)
        # Numărați valorile unice și frecvența lor
        unique_values, counts = np.unique(predictii, return_counts=True)

        # Crearea unui dicționar pentru a cataloga valorile și frecvența lor
        value_counts = {}

        # Iterați prin valorile unice și frecvența lor și adăugați-le în dicționar
        for value, count in zip(unique_values, counts):
            value_counts[value] = count

        maxim = 0
        for key in value_counts:
            if maxim < value_counts[key]:
                maxim = value_counts[key]
                
        msg = ""
        for key in value_counts:
            if value_counts[key] == maxim:
                if msg == "":
                    msg = key
                else:
                    msg += f"/{key}"
                    
        return msg

class Record:
# pe google colab nu va merge PortAudioError: Error querying device -1
# nu exista un device de inregistrare
    def __init__(self, filename, out):
        fs = 44100  # frecventa de esantionare
        max_seconds = 180  # durata maxima inregistrarii scrisa in secunde (180s = 3m)
    
        print("Spune ceva. Apasă 'q' pentru a opri înregistrarea (înregistrarea durează maxim 3 minute)")
        recording = sd.rec(int(max_seconds * fs), samplerate=fs, channels=1)

        start_time = time.time()

        while True:
            elapsed_time = time.time() - start_time
            if keyboard.is_pressed('q') or elapsed_time >= max_seconds:
                sd.stop()
                break

        recording = recording[:int(fs * elapsed_time)]
# ma asigur ca s-a oprit inregistrarea
        sd.wait()
        time.sleep(1)

        sf.write(filename, recording, fs)  # Save as WAV file 
        out.add("Înregistrarea a fost salvată")
        
class App:
    def __init__(self):
        self.version_error = False
        self.trained_models = False
        self.loaded_models = False
        self.from_recorded = False
        self.m = Models()
        self.p = Preprocessing()
        self.files = Files()
        self.mle = My_label_encoder()
        self.recorded_filename = "output.wav"
        self.vgg16 = VGG16_Modif()
        
        if sys.version_info < (3, 6):
            err.add_error("Trebuie să ai minim python 3.6 înainte de a rula această aplicație")
        if sys.version_info >= (3, 10):
            err.add_error("Trebuie să ai sub python 3.10 înainte de a rula această aplicație. Sugerez versiunea 3.9.13")
# Cele mai multe dintre librariile folosite de mine sunt compatibile cu Python 3.x, cu unele necesitand versiunea 3.6 sau 3.7
# Sounddevice, de exemplu, necesită Python 3.6 sau mai recent, iar soundfile necesită Python 3.5 sau mai recent
# Asa ca, pentru ca cineva sa poata rula aplicatia mea, trebuie sa aiba cel putin Python 3.6 instalat

    def train(self, director):
        if not err.got_error():
            # realizare preprocesare pe baza directorului dir
            self.p.pereche_wav_txt(director)
            # extragere date
            # x este pentru fisierele audio
            # y este pentru emotii
            self.x, self.y = self.p.preprocesare()
            # antrenare model
            self.m.train(self.x, self.y)
            # afisare acuratete modele
            self.m.models_accuracy()
            if director.find("emodb/wav") != -1:
                self.m.create_table(self.vgg16.get_accuracy("emodb"))
            else:
                self.m.create_table(self.vgg16.get_accuracy("dataset"))
            self.trained_models = True

    def save_models(self):
        if not self.trained_models:
            err.add_error("Nu poți salva modelele dacă nu au fost antrenate")
        else:
            self.m.save_models()

    def load_models(self):
        self.m.load_models()
        if not err.got_error():
            self.loaded_models = True
  
    def use(self, audio_filename):
        if not err.got_error():
            if not self.trained_models and not self.loaded_models:
                err.add_error("Nu poți folosi modelele dacă nu au fost antrenate sau încărcate de undeva")
            else:
                self.p = Preprocessing()
                em1 = self.m.use_models(audio_filename, self.p)
                em2 = self.vgg16.use(audio_filename, self.mle)
                lista = f"{em1}/{em2}".split("/")
                l = self.vgg16.get_dict_fara_nan(lista)
                em = self.vgg16.vot(l)
                out.add(em)

    def show_output(self):
        if err.got_error():
            print("erori:")
            err.show()
            err.clear()
        if out.have_output():
            print("output:")
            out.show()
            out.clear()
    
    def check_directory(self, directory):
        if directory == "":
            return False
    
        if directory.find("emodb/wav") != -1:
            return True
        
        files = os.listdir(directory)
        count_txt = 0
        count_wav = 0
        count_not_good = 0

        for file in files:
            ext = file[-3:]
            if ext == "wav":
                count_wav += 1
            elif ext == "txt":
                count_txt += 1
            else:
                count_not_good += 1
    
        if count_wav != count_txt or count_not_good != 0:
            return False
        else:
            return True
    
    def start(self, choice):
        if not err.got_error():
            if choice == "antrenare":
                # nu o sa mearga pe colab
                Tk().withdraw() # prevents an empty tkinter window from appearing
                folder = filedialog.askdirectory()
                if self.check_directory(folder):
                    print("Am pornit antrenarea")
                    start_train = datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S")
                    self.train(folder)
                    self.save_models()
                    end_train = datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S")
                    print(f"Am terminat antrenarea (elapsed: {end_train - start_train})")
                else:
                    err.add_error(f"Folderul {folder} nu conține fișiere de forma FILENAME.txt - FILENAME.wav pentru a se putea realiza antrenarea. Alege alt folder")
            elif choice == "folosire":
                self.load_models()
                if not self.trained_models and not self.loaded_models:
                    err.add_error("Nu poți folosi modelele dacă nu au fost antrenate sau încărcate înainte")
                else:
                    r = Record(self.recorded_filename, out)
                    self.use(self.recorded_filename)
            else:
                err.add_error("Alegere neidentificată")

                
# forma adnotarilor: [eticheta emotie] [etichete fundal] [eticheta personaj] text
# eticheta emotie poate avea urmatoarele valori:
#   A = anger
#   B = boredeom
#   D = disgust
#   F = fear or anxiety
#   H = happiness
#   I = irritation or nervousness
#   N = neutral
#   S = sadness
# eticheta fundal poate avea urmatoarele valori:
#   zgomot - pt zona de zgomot
#   zgomot fundal - pt zgomotul de fundal
#   voci - pt voci nedeslusite
#   voci fundal - pentru vocile care se aud pe fundal
#   tipete fundal - pentru vocile de pe fundal care dau impresia de spaima, frica
#   muzica - pt zona de muzica
#   muzica fundal - pt muzica care se aude pe fundal
# eticheta personaj poate avea urmatoarele valori:#
#   apelant - cel care suna la 112
#   operator - cel care raspunde la telefon
#   voci - cel care se aude nu este niciunul dintre cele doua personaje
# text - zona de text reprezentand ceea ce zice personajul


app = App()
done = False

while not done:
    choice = input("1. Antrenare modele\n2. Folosire modele\n3. STOP\nAlegerea ta e: ")
    choice = int(choice)
    alegere = ""
    if choice == 1:
        alegere = "antrenare"
        print("Pentru a realiza antrenarea modelelor, trebuie să selectezi folderul unde ai fișierele text și wav pentru antrenare")
    elif choice == 2:
        alegere = "folosire"
        print("Modelele se vor salva în folderul models")
    else:
        done = True
    
    if choice <= 2:
        app.start(alegere)
        app.show_output()
    
print("Am terminat de rulat aplicația")