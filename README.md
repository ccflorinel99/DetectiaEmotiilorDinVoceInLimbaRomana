# Detectia multimodala a emotiilor din voce in limba romana
Acesta este un repository pentru detectia de emotie din voce in limba romana  
Forma adnotarilor: [eticheta emotie] [etichete fundal] [eticheta personaj] text  
Eticheta emotie poate avea urmatoarele valori:  
 -> A = anger  
 -> B = boredeom  
 -> D = disgust  
 -> F = fear or anxiety  
 -> H = happiness  
 -> I = irritation or nervousness  
 -> N = neutral  
 -> S = sadness  
Eticheta fundal poate avea urmatoarele valori:  
 -> zgomot - pt zona de zgomot  
 -> zgomot fundal - pt zgomotul de fundal  
 -> voci - pt voci nedeslusite  
 -> voci fundal - pentru vocile care se aud pe fundal  
 -> tipete fundal - pentru vocile de pe fundal care dau impresia de spaima, frica  
 -> muzica - pt zona de muzica  
 -> muzica fundal - pt muzica care se aude pe fundal  
Eticheta personaj poate avea urmatoarele valori:  
 -> apelant - cel care suna la 112  
 -> operator - cel care raspunde la telefon  
 -> voci - cel care se aude nu este niciunul dintre cele doua personaje  
Text - zona de text reprezentand ceea ce zice personajul  

# Pentru antrenarea modelelor (am folosit 5 modele)  
Este necesar sa aveti audacity instalat pe laptop/pc, dupa care va trebui sa adnotati fisierul audio (de oricare tip ar fi el si e suportat de audacity) in functie de cine ce vorbeste si sa urmariti etichetele de mai sus.  
La finalul adnorarii, va trebui sa urmati pasii din poza de mai jos si sa exportati 2 tipuri de fisiere din acelasi proiect audacity (odata incarcat un fisier, dati save, salvati-l undeva si va avea extensia .aup3 care va fi considerat proiect de catre audacity): unul .wav si altul .txt  
Pentru cel cu extensia .wav se poate vedea in poza cum se face, iar in cel .txt se vor afla adnotarile, astfel ca va trebui aleasa optiunea "Export Labels"  
**Atentie! Fisierul .wav trebuie sa aiba aceeasi denumire ca cel .txt**  
*Exemplu: FILENAME.wav FILENAME.txt*  
![image](https://user-images.githubusercontent.com/31506258/213907164-4a83bfda-501a-4851-9d4d-7d684f37fdb3.png)
