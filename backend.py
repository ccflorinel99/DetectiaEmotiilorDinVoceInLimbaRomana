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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# type of generalized linear model (GLM) that uses a logistic function to model a binary dependent variable. It estimates the probability that a given input belongs to a particular class and then classifies it based on a threshold (supervised)
from sklearn.tree import DecisionTreeClassifier
# flowchart-like structure, where each internal node represents a feature(or attribute), each branch represents a decision rule, and each leaf node represents the outcome (supervised)
from sklearn.neural_network import MLPClassifier
# type of feedforward artificial neural network that consists of one or more layers of artificial neurons, called perceptrons. Each perceptron receives input from the previous layer and applies a non-linear activation function to it before passing the output to the next layer. The last layer of perceptrons is called the output layer and it generates the final predictions. The algorithm learns the weights of the perceptrons by minimizing the difference between the predicted output and the true output using an optimization algorithm, such as stochastic gradient descent. (supervised)
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier # multiple trees and combines them to make a decision by using "vote system" and the majority wins (supervised)
import os
import pickle
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC # find the best boundary (supervised)
from pydub import AudioSegment
from IPython.display import Audio
from IPython.display import display
import soundfile as sf
import sounddevice as sd
import sys
from scipy.io.wavfile import write
from tkinter import *
from tkinter import filedialog
import keyboard
import time
from pydub import AudioSegment

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
        self.add_error(f"{path} nu exista")

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
            err.add_error(f"Emotia prezisa nu se afla in dictionar, aceasta avand codul {code}")

class AudioManipulation:
    def __init__(self):
        self.files = Files()

    def load(self, file_path):
        if self.files.path_exists(file_path):
            audio = AudioSegment.from_wav(file_path)
              # sr = sample rate
            _ , sr = sf.read(file_path)  
            return audio, sr
# Sample rate, sau "rata de eșantionare", reprezintă numărul de eșantioane de semnal luate pe secundă.
# Este exprimat în herți (Hz) și reprezintă numărul de câți eșantioane sunt luate pe secundă din semnalul audio.
# Rata de eșantionare afectează calitatea sunetului și frecvența maximă care poate fi înregistrată sau redată.
# Cu cât rata de eșantionare este mai mare, cu atât semnalul audio este redat mai precis.
# O rata de eșantionare mai mare va permite redarea frecvențelor mai înalte, dar va ocupa mai mult spațiu pe disc.
# Rata standard pentru aplicațiile audio digitale este de 44.1 sau 48 kHz.
        else:
            err.path_not_found(file_path)

# STFT (Short-Time Fourier Transform) este o metodă utilizată pentru a analiza un semnal în timp și frecvență. Aceasta se face prin împărțirea semnalului în fragmente scurte de durată (numite "ferestre") și aplicarea unei transformări Fourier pe fiecare fragment. Aceasta permite analizarea semnalului în timp și frecvență simultan.
# In acest caz, se utilizeaza functia numpy hanning pentru a obtine fereastra hanning si se prelungește audio cu câteva esantioane pentru a evita pierderea acestora la tăiere. Apoi, se utilizeaza functia fft din numpy pentru a efectua transformata Fourier rapida, si se returneaza spectrograma absoluta.
    def stft(self, audio, window_size=2048):
        audio_len = len(audio)
        window = np.hanning(window_size)
# https://en.wikipedia.org/wiki/Hann_function
# Operatorul // este operatorul de diviziune care returneaza catul diviziei, eliminand partea fractionara.
        audio = np.pad(audio, (window_size//2, window_size//2), mode='reflect')
        num_segments = (audio_len + window_size - 1) // window_size
        segments = np.zeros((num_segments, window_size))
# Fereastra este o functie matematica care se aplica peste segmentul audio si are rolul de a reduce artefactele care apar in urma taierii audio-ului in segmente.
# Aceasta reduce artefactele prin atenuarea (attenuation) marginilor segmentelor.
# artefacte = distorsiuni sau abateri neintentionate in semnalul audio care apar in urma procesarii acestuia
        for i in range(num_segments):
            segment = audio[i*window_size:i*window_size+window_size]
            segments[i] = segment * window
        spectrogram = np.abs(np.fft.fft(segments, axis=1))
# Aceasta linie de cod calculeaza spectrograma semnalului audio prin utilizarea Transformatei Fourier Rapide (FFT). Spectrograma este o reprezentare grafica a semnalului audio, care arata cum se distribuie energia in diferite frecvente de-a lungul timpului.
# Fiecare coloana a spectrogram-ului reprezinta o transformata Fourier a unui segment din semnalul audio, iar fiecare linie reprezinta o frecventa specifica.
# Se ia modulul FFT pentru a elimina componenta de faza.
# Acesta este utilizat pentru a detecta si caracteriza anumite caracteristici ale sunetului cum ar fi tonalitatea sau timbrul.
        return spectrogram

    def piptrack(self, audio, sr):
    # calculam diferenta de faza intre semnalul original si cel intarziat
        audio_delayed = audio[1:]
        audio_diff = audio_delayed - audio[:-1]
        audio_diff = np.square(audio_diff)

    # aplicam o fereastra de Hanning
        window = np.hanning(len(audio_diff))
        audio_diff = audio_diff * window

    # calculam integrala spectrului
        audio_diff = np.cumsum(audio_diff)

    # determinam pitch-ul
        pitch = 0
        if np.argmin(audio_diff) != 0:
            pitch = sr / np.argmin(audio_diff)

        return pitch
# Aceasta functie calculeaza rata de variatie a frecventei vorbirii (pitch) din semnalul audio.
# Prima etapa este calcularea diferentei de faza intre semnalul original si cel intarziat.
# Aceasta este realizata prin scaderea elementelor din semnalul original cu elementele din semnalul intarziat.
# Urmatorul pas este ridicarea la patrat a diferentei de faza pentru a elimina valorile negative.
# Apoi se aplica o fereastra de Hanning pentru a reduce artefactele.
# Se calculeaza integrala spectrului, care este utilizata pentru a determina pozitia minimului, care este inversul frecventei.
# Rata de variatie a frecventei vorbirii (pitch) este obtinuta prin impartirea frecventei de esantionare la pozitia minimului.

class Preprocessing():
    def __init__(self):
        self.files = Files()
        self.am = AudioManipulation()
        
    def pereche_wav_txt(self, dir : str):
        self.dir = dir
        if not self.files.path_exists(dir): # verifica daca directorul exista sau nu
            err.path_not_found(dir)
        else:
      # listam fisierele din directorul dat ca parametru
            filenames = os.listdir(self.dir)

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
            out.add("Ai de rezolvat una sau mai multe erori inainte de a lucra cu functia preprocess_audio_file")
            return [0], 0
        else: 
      # incarcam fisierul audio
            audio, sr = self.am.load(audio_file)
            audios = []
    
            if(self.files.path_exists(txt_file)):
                with open(txt_file, 'r') as f:
                    lines = f.readlines()
                    out.add(f"txt file: {txt_file}, nr linii: {len(lines)}")
                    # impartim fisierul audio in fragmente
                    for line in lines:
                        start = float(line[:line.find("[")].split(' ')[0].split('\t')[0])
                        end = float(line[:line.find("[")].split(' ')[0].split('\t')[1])
                        fragment_audio = audio[start:end]
                        out.add(f"range {start} - {end}")
                        # adaugam fragmentul audio la lista de fragmente
                        audios.append(fragment_audio)
                    f.close()
            else:
                err.path_not_found(txt_file)
    
        return audios, sr

    def extract_features(self, audio, sr):
        if err.got_error():
            out.add("Ai de rezolvat una sau mai multe erori inainte de a lucra cu functia extract_features")
            return [0, 0, 0, 0, 0, 0]
        else: 
            # transformare din pydub audiosegment in numpy array
            audio = np.asarray(audio.get_array_of_samples(),dtype = np.float64)
            # calculam amplitudinea semnalului
            amplitudes = np.abs(audio)

            # calculam spectrul de frecventa
            spectrogram = np.abs(self.am.stft(audio))

            # calculam rata de variatie a frecventei vorbirii
            pitch = self.am.piptrack(audio, sr)
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
            a = [np.mean(amplitudes), np.mean(spectrogram), pitch_change_rate, np.mean(syllable_lengths), np.mean(pause_lengths), respiration_rate]
            # pitch_change_rate si respiration_rate au cele mai mari sanse sa dea nan  
            a = np.nan_to_num(a)
            
            count = len(a)
            
            suma = 0
            for i in range(count):
                suma = suma + a[i]
                
            a = [suma/count]
            a = np.array(a)
            return a

  
    def preprocesare(self):
        if err.got_error():
            out.add("Ai de rezolvat una sau mai multe erori inainte de a lucra cu functia preprocesare")
            return 0, 0
        else: 
            # obtinem listele audio_files si txt_files
            audio_files = [pair[0] for pair in self.audio_txt_pairs]
            txt_files = [pair[1] for pair in self.audio_txt_pairs]

            # citim etichetele de emotie din fisierele txt
            labels = []
            for txt_file in txt_files:
                with open(txt_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        label = line[line.find("["):].split(' ')[0].replace("[", "").replace("]", "")
                        labels.append(label)

            # extragem caracteristicile din fisierele audio
            X = []

            for i in range(len(self.audio_txt_pairs)):
                audio, sr = self.preprocess_audio_file(audio_files[i], txt_files[i])
                for j in range(len(audio)):
                    features = self.extract_features(audio[j], sr)
                    X = np.append(X, features)

            out.add(f"{txt_files[i]}, X = {X}")
    
            # transformam etichetele de emotie in forma utilizabila de model
            le = LabelEncoder()
            y = le.fit_transform(labels)
            out.add(f"y = {y}")

            # Elimină lista suplimentară din interiorul lui np.array() atunci când se definește X
            X = np.resize(X, (y.shape[0], 1))
      
            return X, y


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
        self.MLPClassifierModel = MLPClassifier(max_iter=5000)
# dadea o eroare ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
# rezolvare: https://www.google.com/search?q=ConvergenceWarning%3A+Stochastic+Optimizer%3A+Maximum+iterations+(200)+reached+and+the+optimization+hasn%27t+converged+yet.&rlz=1C1CHBF_enRO857RO857&oq=ConvergenceWarning%3A+Stochastic+Optimizer%3A+Maximum+iterations+(200)+reached+and+the+optimization+hasn%27t+converged+yet.&aqs=chrome..69i57j69i58.721j0j1&sourceid=chrome&ie=UTF-8
        self.files = Files()

    def save_models(self):
        if err.got_error():
            out.add("Ai de rezolvat una sau mai multe erori inainte de a salva modelele")
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
            err.add_error("Folderul 'models' nu exista, rezulta faptul ca va trebui sa antrenezi intai modelele")
        else:
            count = 0
            if not self.files.path_exists("models/RandomForestClassifierModel.pkl"):
                err.add_error("Fisierul 'models/RandomForestClassifierModel.pkl' nu exista")
                count = count + 1
            if not self.files.path_exists("models/SVCModel.pkl"):
                err.add_error("Fisierul 'models/SVCModel.pkl' nu exista")
                count = count + 1
            if not self.files.path_exists("models/LogisticRegressionModel.pkl"):
                err.add_error("Fisierul 'models/LogisticRegressionModel.pkl' nu exista")
                count = count + 1
            if not self.files.path_exists("models/DecisionTreeClassifierModel.pkl"):
                err.add_error("Fisierul 'models/DecisionTreeClassifierModel.pkl' nu exista")
                count = count + 1
            if not self.files.path_exists("models/MLPClassifierModel.pkl"):
                err.add_error("Fisierul 'models/MLPClassifierModel.pkl' nu exista")
                count = count + 1
                
            if count == 5: # daca nu exista niciun model
                err.add_error("Nu am gasit niciun model, asa ca va trebui sa antrenezi modelele")
                
        if not err.got_error():
            with open("models/RandomForestClassifierModel.pkl", "rb") as f:
                self.RandomForestClassifierModel = pickle.load(f)
            out.add("models/RandomForestClassifierModel a fost incarcat")

            with open("models/SVCModel.pkl", "rb") as f:
                self.SVCModel = pickle.load(f)
            out.add("models/SVCModel a fost incarcat")

            with open("models/LogisticRegressionModel.pkl", "rb") as f:
                self.LogisticRegressionModel = pickle.load(f)
            out.add("models/LogisticRegressionModel a fost incarcat")

            with open("models/DecisionTreeClassifierModel.pkl", "rb") as f:
                self.DecisionTreeClassifierModel = pickle.load(f)
            out.add("models/DecisionTreeClassifierModel a fost incarcat")

            with open("models/MLPClassifierModel.pkl", "rb") as f:
                self.MLPClassifierModel = pickle.load(f)
            out.add("models/MLPClassifierModel a fost incarcat")

    def train(self, x, y):
        if err.got_error():
            out.add("Nu poti antrena modelele daca ai erori")
        elif x.size == 0 and y.size == 0:
            err.add_error("Nu s-a realizat preprocesarea")
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.7, random_state=42) # 70% date de antrenare, 30% date de testare
# random_state = 42 means that the split is reproducible, meaning that if the script is run again with the same random_state, the data will be split in the same way
            out.add(f"X_train: {self.X_train}")
            out.add(f"y_train: {self.y_train}")
            out.add(f"X_test: {self.X_test}")
            out.add(f"y_test: {self.y_test}")
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

            out.add(f'Acuratetea modelului RandomForestClassifier este: {self.accuracy_RandomForestClassifier:.2f}')
            out.add(f'Acuratetea modelului SVC este: {self.accuracy_SVC:.2f}')
            out.add(f'Acuratetea modelului LogisticRegression este: {self.accuracy_LogisticRegression:.2f}')
            out.add(f'Acuratetea modelului DecisionTreeClassifier este: {self.accuracy_DecisionTreeClassifier:.2f}')
            out.add(f'Acuratetea modelului MLPClassifier este: {self.accuracy_MLPClassifier:.2f}')

    def create_table(self):
        from tabulate import tabulate

        #create data
        data = [["RandomForestClassifier Model", self.accuracy_RandomForestClassifier], 
                ["SVC Model", self.accuracy_SVC], 
                ["LogisticRegression Model", self.accuracy_LogisticRegression], 
                ["DecisionTreeClassifier Model", self.accuracy_DecisionTreeClassifier],
                ["MLPClassifier Model", self.accuracy_MLPClassifier]]
  
        #define header names
        col_names = ["Models", "Accuracy"]
  
        #display table
        out.add("\n")
        out.add(tabulate(data, headers=col_names))


    def use_models(self, audio_filename, p : Preprocessing):
        audio, sr = self.am.load(audio_filename)
        self.predictii = [] # aici voi stoca predictiile

        # extrageti caracteristicile fisierului audio
        X = p.extract_features(audio, sr).reshape(-1, 1)
        
        # folositi modelele pentru a face predictia
        self.prediction_RandomForestClassifierModel = self.RandomForestClassifierModel.predict(X)
        self.prediction_SVCModel = self.SVCModel.predict(X)
        self.prediction_LogisticRegressionModel = self.LogisticRegressionModel.predict(X)
        self.prediction_DecisionTreeClassifierModel = self.DecisionTreeClassifierModel.predict(X)
        self.prediction_MLPClassifierModel = self.MLPClassifierModel.predict(X)

        # transformati rezultatul predictiei in eticheta de emotie
        label_encoder = LabelEncoder()
        emotion_labels = label_encoder.fit_transform(self.emotions.to_list())
        out.add(f"Codificari: {emotion_labels}")
        out.add(f"Emotii: {self.emotions.to_list()}")
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
        
        out.add(f"Predictia lui RandomForestClassifierModel este {self.emotion_RandomForestClassifierModel} ({emotion_labels[self.prediction_RandomForestClassifierModel[0]]})")
        out.add(f"Predictia lui SVCModel este {self.emotion_SVCModel} ({emotion_labels[self.prediction_SVCModel[0]]})")
        out.add(f"Predictia lui LogisticRegressionModel este {self.emotion_LogisticRegressionModel} ({emotion_labels[self.prediction_LogisticRegressionModel[0]]})")
        out.add(f"Predictia lui DecisionTreeClassifierModel este {self.emotion_DecisionTreeClassifierModel} ({emotion_labels[self.prediction_DecisionTreeClassifierModel[0]]})")
        out.add(f"Predictia lui MLPClassifierModel este {self.emotion_MLPClassifierModel} ({emotion_labels[self.prediction_MLPClassifierModel[0]]})")
        out.add(f"Decizia finala este {self.predictie()}")
        
        
    def predictie(self):
        counts = [] # unde pun fiecare de cate ori apare
        em = [] #unde o sa pun emotiile
    
        for key in self.emotions.d:
            emotie = self.emotions.d[key]
            count = 0
            for i in range(len(self.predictii)):
                if self.predictii[i] == emotie:
                    count += 1
            
            counts.append(count)
            em.append(emotie)
        
        maxim = 0
        minim = 0
        e = "" # emotia finala
    
        for i in range(len(counts)):
            if counts[i] == minim:
                pass
            elif counts[i] > maxim:
                maxim = counts[i]
        
        ctr = 0 # pt egale
        for i in range(len(counts)):
            if counts[i] == maxim:
                ctr += 1
                if e == "":
                    e = self.emotions.d[i]
                else:
                    e = e + "/" + self.emotions.d[i]
                
        return e

class Record:
# pe google colab nu va merge PortAudioError: Error querying device -1
# nu exista un device de inregistrare
    def __init__(self, filename):
        fs = 44100  # frecventa de esantionare
        max_seconds = 180  # durata maxima inregistrarii scrisa in secunde (180s = 3m)
    
        print("Spune ceva. Apasa 'q' pentru a opri inregistrarea (inregistrarea dureaza maxim 3 minute)")
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
        print("Inregistrarea a fost salvata")


class App:
    def __init__(self):
        self.version_error = False
        self.trained_models = False
        self.loaded_models = False
        self.from_recorded = False
        self.m = Models()
        self.p = Preprocessing()
        self.files = Files()
        self.recorded_filename = "output.wav"
        if sys.version_info < (3, 6):
            err.add_error("Trebuie sa ai minim python 3.6 inainte sa poti rula aceasta aplicatie")
# Cele mai multe dintre librariile folosite de mine sunt compatibile cu Python 3.x, cu unele necesitand versiunea 3.6 sau 3.7
# Sounddevice, de exemplu, necesită Python 3.6 sau mai recent, iar soundfile necesită Python 3.5 sau mai recent
# Asa ca, pentru ca cineva sa poata rula aplicatia mea, trebuie sa aiba cel putin Python 3.6 instalat

    def train(self, dir):
        if not err.got_error():
            # realizare preprocesare pe baza directorului dir
            self.p.pereche_wav_txt(dir)
            # extragere date
            # x este pentru fisierele audio
            # y este pentru emotii
            self.x, self.y = self.p.preprocesare()
            # antrenare model
            self.m.train(self.x, self.y)
            # afisare acuratete modele
            self.m.models_accuracy()
            # creare tabel pentru a putea vedea detaliile mai bine
            self.m.create_table()
            self.trained_models = True

    def save_models(self):
        if not self.trained_models:
            err.add_error("Nu poti salva modelele daca nu au fost antrenate")
        else:
            self.m.save_models()

    def load_models(self):
        self.m.load_models()
        if not err.got_error():
            self.loaded_models = True
  
    def use(self, audio_filename):
        if not err.got_error():
            if not self.trained_models and not self.loaded_models:
                err.add_error("Nu poti folosi modelele daca nu au fost antrenate sau incarcate de undeva")
            else:
                self.p = Preprocessing()
                self.m.use_models(audio_filename, self.p)

    def show_output(self):
        if err.got_error():
            print("erori:")
            err.show()
            err.clear()
        if out.have_output():
            print("output:")
            out.show()
            out.clear()

    def start(self, choice):
        if not err.got_error():
            if choice == "antrenare":
                # nu o sa mearga pe colab
                Tk().withdraw() # prevents an empty tkinter window from appearing
                dir = filedialog.askdirectory()
                self.train(dir)
                self.save_models()
            elif choice == "folosire":
                self.load_models()
                if not self.trained_models and not self.loaded_models:
                    err.add_error("Nu poti folosi modelele daca nu au fost antrenate sau incarcate inainte")
                else:
                    r = Record(self.recorded_filename)
                    self.use(self.recorded_filename)
            else:
                err.add_error("Alegere neidentificata")


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
        print("Pentru a realiza antrenarea modelelor, trebuie sa selectezi folderul unde ai fisierele text si wav pentru antrenare")
    elif choice == 2:
        alegere = "folosire"
        print("Modelele se vor salva in folderul models")
    else:
        done = True
    
    if choice <= 2:
        app.start(alegere)
        app.show_output()
    
print("Am terminat de rulat aplicatia")

