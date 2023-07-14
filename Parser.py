import re
import os
import pandas as pd

index_start = 0
index_end = 1
index_eticheta_emotie = 3
index_eticheta_fundal = 5
index_eticheta_personaj = 7


def regex_etichete(s):
    x = re.split("[\[\]\t]", s)
    return x
        
def e_fisier(filepath):
    if os.path.exists(filepath) and not os.path.isdir(filepath):
        return True
    else:
        return False
        
def probleme_cu_calea(filepath):
    message = ""
    if not os.path.exists(filepath):
        print(f"Calea {filepath} este invalida")
        message = f"Calea {filepath} este invalida"
    else:
        if not os.path.isdir(filepath):
            print(f"Calea {filepath} reprezinta un director sau fisier special (socket, FIFO, etc)")
            message = f"Calea {filepath} reprezinta un director sau fisier special (socket, FIFO, etc)"
            
    return message
    
def all_ok(filepath1, filepath2, filepath3):
    if e_fisier(filepath1) and e_fisier(filepath2) and e_fisier(filepath3):
        return True
    else:
        return False
    
def e_emotie(em):
    regex = re.search("[ABDFHSNI]", em)
    if regex:
        return True
    else:
        return False
    
    
def nr_linii(filename):
    f = open(filename, "r")
    lines = f.readlines()
    f.close()
    return len(lines)
        

def verifica(counter1, counter2, counter3):
    if(counter1 == counter2 and counter2 == counter3):
        return True
    else:
        return False

def concatenare_mesaje_erori(message1, message2, message3):
    message = ""
    if message1 != "":
        message = message + "; " + message1
        
    if message2 != "":
        message = message + "; " + message2
        
    if message3 != "":
        message = message + "; " + message3
        
    return message
     
def vot(var1, var2, var3):
    decizie = ""
    if var1 == var2 and var1 == var3:
        decizie = var1
    elif (var1 == var2 and var1 != var3) or (var1 == var3 and var1 != var2):
        decizie = var1
    elif var2 == var3 and var1 != var2:
        decizie = var3
    else:
        decizie = var1 + "/" + var2 + "/" + var3
        
    return decizie
    
    
def write_excel(filename1, filename2, filename3, outfile):
    f1 = open(filename1, "r")
    f2 = open(filename2, "r")
    f3 = open(filename3, "r")
    o = open("Rezultat.txt", "w")
    counter = nr_linii(filename1)
    emotii, fundal, personaj = [], [], []
    start_end = []
    decizii = []
    
    for i in range(counter):
        line1 = f1.readline() # linia din f1
        line2 = f2.readline() # linia din f2
        line3 = f3.readline() # linia din f3
        re1 = regex_etichete(line1)
        re2 = regex_etichete(line2)
        re3 = regex_etichete(line3)
        #print(re3)
        
        start1, end1 = re1[index_start], re1[index_end]
        start2, end2 = re2[index_start], re2[index_end]
        start3, end3 = re3[index_start], re3[index_end]
        decizie_start, decizie_end = vot(start1, start2, start3), vot(end1, end2, end3)
        
        em1, em2, em3 = re1[index_eticheta_emotie], re2[index_eticheta_emotie], re3[index_eticheta_emotie]
        decizie_em = vot(em1, em2, em3)
        
        fundal1, fundal2, fundal3 = re1[index_eticheta_fundal], re2[index_eticheta_fundal], re3[index_eticheta_fundal]
        decizie_fundal = vot(fundal1, fundal2, fundal3)
        
        personaj1, personaj2, personaj3 = re1[index_eticheta_personaj], re2[index_eticheta_personaj], re3[index_eticheta_personaj]
        decizie_personaj = vot(personaj1, personaj2, personaj3)
        
        start_end.append([f"{start1}-{end1}", f"{start2}-{end2}", f"{start3}-{end3}", f"{decizie_start}-{decizie_end}"])
        emotii.append([em1, em2, em3, decizie_em])
        fundal.append([fundal1, fundal2, fundal3, decizie_fundal])
        personaj.append([personaj1, personaj2, personaj3, decizie_personaj])
        decizii.append([f"{decizie_start}-{decizie_end}", decizie_em, decizie_fundal, decizie_personaj])
        start_end_txt = f"[{decizie_start}-{decizie_end}]"
        emotie_txt = f"[{decizie_em}]"
        fundal_txt = f"[{decizie_fundal}]"
        personaj_txt = f"[{decizie_personaj}]"
        text_txt = f"{start_end_txt} {emotie_txt} {fundal_txt} {personaj_txt}"
        if(i != counter - 1):
            text_txt = text_txt + "\n"
        o.write(text_txt)
        
        
    f1.close()
    f2.close()
    f3.close()
    o.close()
    
    # Setarea opțiunilor de afișare pentru pandas
    pd.set_option('display.max_rows', None)  # Afișează toate rândurile
    pd.set_option('display.max_columns', None)  # Afișează toate coloanele
    
    index = [i+1 for i in range(len(start_end))]
    col_start_end = ['start-end v1', 'start-end v2', 'start-end v3', 'decizie']
    col_emotii = ['emotie v1', 'emotie v2', 'emotie v3', 'decizie']
    col_fundal = ['fundal v1', 'fundal v2', 'fundal v3', 'decizie']
    col_personaj = ['personaj v1', 'personaj v2', 'personaj v3', 'decizie']
    col_decizii = ['decizie_start_end', 'decizie_emotie', 'decizie_fundal', 'decizie_personaj']
    
    df1 = pd.DataFrame(start_end, index=index, columns=col_start_end)
    df2 = pd.DataFrame(emotii, index=index, columns=col_emotii)
    df3 = pd.DataFrame(fundal, index=index, columns=col_fundal)
    df4 = pd.DataFrame(personaj, index=index, columns=col_personaj)
    combined_df = pd.concat([df1, df2, df3, df4], axis=1)
    #print(combined_df)
    #combined_df.to_excel(outfile, sheet_name='Sheet1')
    df = pd.DataFrame(decizii, index=index, columns=col_decizii)
    print(df)
    df.to_excel(outfile, sheet_name='Sheet1')

def etichete_egale(filename1, filename2, filename3, outfile):
     write_excel(filename1, filename2, filename3, outfile)
        

def script():
    filepath1 = input("Introduceti numele fisierului corespunzator primei maini: ")
    filepath2 = input("Introduceti numele fisierului corespunzator mainii a doua: ")
    filepath3 = input("Introduceti numele fisierului corespunzator mainii a treia: ")
    outfile = "Rezultat.xlsx"

    # decomenteaza urmatoarea linie daca vrei sa iti alegi singur ce nume sa aiba fisierul de output
    #outfile = input("Introdu numele fisierului de output:")
    
    if all_ok(filepath1, filepath2, filepath3):
        counter1 = nr_linii(filepath1)
        counter2 = nr_linii(filepath2)
        counter3 = nr_linii(filepath3)
    
        rez = verifica(counter1, counter2, counter3)

        if rez:
            etichete_egale(filepath1, filepath2, filepath3, outfile)
        else:
            print("EROARE! Fisierele au diferite adnotari. Varianta finala trebuie facuta de tine")
    else:
        message1 = probleme_cu_calea(filepath1)
        message2 = probleme_cu_calea(filepath2)
        message3 = probleme_cu_calea(filepath3)
    
        message = concatenare_mesaje_erori(message1, message2, message3)
        print(message)
    

    print(f"Fisierul {outfile} a fost scris")
    
    
script()