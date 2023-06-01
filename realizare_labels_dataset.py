import os
from datetime import datetime
import librosa
import re
import pickle



index_start = 0
index_end = 1
index_eticheta_emotie = 3


def regex_etichete(s):
    x = re.split("[\[\]\t]", s)
    return x
    
    
def nr_linii(filename):
    f = open(filename, "r")
    lines = f.readlines()
    f.close()
    return len(lines)


    
def get_emotions(filename):
    f = open(filename, "r")
    counter = nr_linii(filename)
    emotii, start_end = [], []
    
    for i in range(counter):
        line = f.readline()
        re = regex_etichete(line)
        
        start, end = re[index_start], re[index_end]
        em = re[index_eticheta_emotie]
        start_end.append(f"{start}-{end}")
        emotii.append(em)
        
        
    f.close()
    
    return start_end, emotii

def start_end_emotii(filename_etichetare):
    se_emotii = {}

    start_end, emotii = get_emotions(filename_etichetare)
    for pereche in zip(start_end, emotii):
        se_emotii[pereche[0]] = pereche[1]
        
    return se_emotii

def gaseste_interval(start, end, dictionar):
    index = 0
    for key in dictionar:
        start2, end2 = key.split("-")
        start2, end2 = round(float(start2), 2), round(float(end2), 2)
        if start >= start2 and end <= end2:
            return index
        else:
            index += 1
            
    return -1 # nu a fost gasit intervalul


def val_at_index(index, dictionar):
    if index == -1 or index == len(dictionar):
        return ""
    else:
        i = 0
        for key in dictionar:
            if i == index:
                return dictionar[key]
            else:
                i += 1
                

def perechi_se(dicitonar, eticheta_end):
    se_perechi = {}

    i = 1
    elem = ""
    for key in dicitonar:
        start, end = key.split("-")
        start, end = round(float(start), 1), round(float(end), 1)
        if i % 2 == 0:
            se_perechi[elem] = f"{start}-{end}"
            elem = ""
        else:
            elem = f"{start}-{end}"
        i += 1

    if not elem == "":
        se_perechi[elem] = eticheta_end
        
    return se_perechi


def round_dictionar(dictionar):
    new_dict = {}
    
    for key in dictionar:
        start, end = key.split("-")
        start, end = round(float(start), 1), round(float(end), 1)
        new_dict[f"{start}-{end}"] = dictionar[key]
    
    return new_dict

def nearest_neighbor(start, end, round_dictionar):
    diferente_start, diferente_end = [], []
    
    for key in round_dictionar:
        start2, end2 = key.split("-")
        start2, end2 = float(start2), float(end2)
        diferente_start.append(start - start2)
        diferente_end.append(end - end2)
        
    return diferente_start, diferente_end

def minim_pozitiv_lista(lista):
    minim = 999
    i = 0
    index = -1
    for elem in lista:
        if minim > elem and elem >=0:
            minim = elem
            index = i
        i += 1
            
    return index

def anomalie(start, end, se_perechi, eticheta_end, round_dict):
    for key in se_perechi:
        start2, end2 = key.split("-")
        start2, end2 = float(start2), float(end2)
        if se_perechi[key] != eticheta_end:
            start3, end3 = se_perechi[key].split("-")
            start3, end3 = float(start3), float(end3)
            if start >= start2 and end >= start3:
                diferente_start, diferente_end = nearest_neighbor(start, end, round_dict)
                index_start = minim_pozitiv_lista(diferente_start)
                index_end = minim_pozitiv_lista(diferente_end)
                em_start = val_at_index(index_start, round_dict)
                em_end = val_at_index(index_end, round_dict)
                if em_start == em_end:
                    return em_start
                else:
                    if em_start == "":
                        return em_end
                    elif em_end == "":
                        return em_start
                    else:
                        return f"{em_start}/{em_end}"
            
        else:
            if start >= start2 and end >= end2:
                print(f"Anomalie la final {start}-{end}")
                return f"Anomalie la final {start}-{end}"
        
def alegere(start_end, dictionar):
    start, end = start_end.split("-")
    start, end = float(start), float(end)
    index = gaseste_interval(start, end, dictionar)
    eticheta_end = "END"
    round_dict = round_dictionar(dictionar)
    se_perechi = perechi_se(dictionar, eticheta_end)
    if index == -1: # poate sa fie anomalie
        val = anomalie(start, end, se_perechi, eticheta_end, round_dict)
        if val == None: # daca nu e anomalie si pur si simplu nu e emotie acolo
            val = ""
    else:
        val = val_at_index(index, dictionar)
    return val


def salveaza(outfile, lista):
    writer = open(outfile, 'wb')
    pickle.dump(lista, writer)
    writer.close()

def progres(folder_labels):
    labels = len(os.listdir(folder_labels)) - 42 # o sa le bag in acelasi folder cu spectrogramele
    p = round(labels * 100 / 42, 2)
    return f"Progres: {p}%"


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


for nr_fisier in range(1):
    nr = 42#nr_fisier + 1
    outfile = "spectrograme/" + str(nr) + ".pkl"
    audio_file = "dataset/" + str(nr) + ".wav"
    filename_etichetare = audio_file[:-3] + "txt"

    audio, sr = librosa.load(audio_file)

    # Calculați lungimea fiecărui cadru în eșantioane
    frame_length = int(sr)
    hop_length = int(sr * 0.01)

    # Segmentați fișierul audio în cadre de 1 secundă cu un pas de 10 ms
    audio_frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    #print(audio_frames.shape)

    alegeri = []
    se_emotii = start_end_emotii(filename_etichetare)

    i = 0
    print(f"frames: {len(audio_frames.T)}")
    file_start_time = datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S")
    for frame in audio_frames.T:
        step = i * 0.01
        start, end = 0 + step, 1 + step
        #print(frame.shape)
        alegeri.append(alegere(f"{start}-{end}", se_emotii))
        i += 1
        if i == round(len(audio_frames.T) / 2):
            current_time = datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S")
            print(f"Halfway there. Elapsed untill now {current_time - file_start_time}")
        
    file_end_time = datetime.strptime(datetime.now().strftime("%H:%M:%S"), "%H:%M:%S")
    mle = My_label_encoder()
    alegeri = mle.encode_list(alegeri)
    salveaza(outfile, alegeri)
    print(f"done ({file_end_time - file_start_time}), {progres('spectrograme')}")

