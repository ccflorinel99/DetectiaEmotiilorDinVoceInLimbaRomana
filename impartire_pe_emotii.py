from pydub import AudioSegment
import os

class Impartire_pe_emotii:
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
        
        self.foldere = {
            "A_folder" : "",
            "B_folder" : "",
            "D_folder" : "",
            "F_folder" : "",
            "H_folder" : "",
            "I_folder" : "",
            "N_folder" : "",
            "S_folder" : ""
        }
        self.folder_emotii = "emotii"
        self.creeaza_folder(self.folder_emotii)
        
        for key_dict, key_folder in zip(self.dict, self.foldere):
            folder = self.folder_emotii + "/" + key_dict
            self.creeaza_folder(folder)
            self.foldere[key_folder] = folder
            
    def creeaza_folder(self, nume):
        if not os.path.exists(nume):
            os.mkdir(nume)
            
    def get_folder(self, emotie):
        return list(self.foldere.values())[self.dict[emotie]]
    
    def get_no_of_file(self, folder):
        return len(os.listdir(folder)) + 1
    
    def imparte(self, file_path):
        audio = AudioSegment.from_wav(file_path)
        # sr = sample rate
        sr = audio.frame_rate
        txt_file = file_path[:-3] + "txt"
        with open(txt_file, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            # impartim fisierul audio in fragmente
            for line in lines:
                start = float(line[:line.find("[")].split(' ')[0].split('\t')[0]) * 1000
                end = float(line[:line.find("[")].split(' ')[0].split('\t')[1]) * 1000
                fragment_audio = audio[start:end]
                em = line.split("\t")[2].split("[")[1].split("]")[0]
                folder = self.get_folder(em)
                fisier = f"{folder}/output{str(self.get_no_of_file(folder))}.wav"
                fragment_audio.export(fisier, format="wav")
                
            f.close()
        
i = Impartire_pe_emotii()
#i.imparte("dataset_sts/2 degete taiate_Final_34sec_Paul_Andrei_Florina_FINAL.wav")
path = "dataset"
files = os.listdir(path)
counter = 1
for file in files:
    ext = file[-3:]
    if ext == "wav":
        i.imparte(path + "/" + file)
        print(f"done {counter}/42")
        counter += 1
print("done")
