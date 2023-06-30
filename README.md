# Detecția multimodală a emoțiilor din voce în limba română
Acesta este un repository pentru detecția de emoții din voce în limba română  
Forma adnotarilor: [eticheta emoție] [etichete fundal] [eticheta personaj] text  
Eticheta emoție poate avea următoarele valori:  
 -> A = anger  
 -> B = boredeom  
 -> D = disgust  
 -> F = fear or anxiety  
 -> H = happiness  
 -> I = irritation or nervousness  
 -> N = neutral  
 -> S = sadness  
Eticheta fundal poate avea următoarele valori:  
 -> zgomot - pt zona de zgomot  
 -> zgomot fundal - pt zgomotul de fundal  
 -> voci - pt voci nedeslușite  
 -> voci fundal - pentru vocile care se aud pe fundal  
 -> tipete fundal - pentru vocile de pe fundal care dau impresia de spaimă, frică  
 -> muzica - pt zona de muzică  
 -> muzica fundal - pt muzică care se aude pe fundal  
Eticheta personaj poate avea următoarele valori:  
 -> apelant - cel care sună la 112  
 -> operator - cel care raspunde la telefon (operatorul de la 112)  
 -> voci - cel care se aude nu este niciunul dintre cele două personaje  
Text - zona de text reprezentând ceea ce zice personajul  

# Modele folosite  
Versiunea 1:  
RandomForestClassifier  
SVC  
LogisticRegression  
DecisionTreeClassifier  
MLPClassifier  
Versiunea 2:  
VGG16, având ultimul layer înlocuit de mine (vezi în cod)  

# Pentru antrenarea modelelor  
Este necesar să aveți audacity instalat pe laptop/pc, după care va trebui să adnotați fișierul audio (de oricare tip ar fi el și e suportat de audacity) în funcție de cine ce vorbește și să aveți în vedere ca etichetărilea realizate de dumneavoastră să fie conform specificațiilor de mai sus.  
La finalul adnotării, va trebui să urmati pașii din poza de mai jos și sa exportați 2 tipuri de fișiere din același proiect audacity (odată încărcat un fișier, dați save, salvați-l undeva și va avea extensia .aup3 care va fi considerat proiect de către audacity): unul .wav și altul .txt.  
Pentru cel cu extensia .wav se poate vedea în poză cum se face, iar în cel .txt se vor afla adnotările, astfel că va trebui aleasă opțiunea "Export Labels".  
**Atentie! Fișierul .wav trebuie să aibă aceeași denumire ca cel .txt**  
*Exemplu: FILENAME.wav FILENAME.txt*  
![image](https://user-images.githubusercontent.com/31506258/213907164-4a83bfda-501a-4851-9d4d-7d684f37fdb3.png)  

Pentru a putea folosi această aplicație aveți nevoie de câteva pachete instalate. Dacă nu le aveți, puteți rula comanda ```pip install numpy pandas scikit-learn opencv-python soundfile sounddevice pydub librosa tensorflow matplotlib keyboard tabulate pysimplegui``` sau dacă aveți anaconda, rulați comanda ```conda install numpy pandas scikit-learn opencv-python soundfile sounddevice pydub librosa tensorflow matplotlib keyboard tabulate pysimplegui```
