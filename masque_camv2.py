import numpy as np
import cv2


#Classificateurs contiennent des données sur les visages etc.. pour apprendre à la machine à les reconnaitre
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mouth.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_nose.xml')


c=0
ok=0
cap = cv2.VideoCapture(0)   #0 : Webcam intégrée
mouths=[]
noses=[]
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)            	#Convertis la vidéo en noir/blanc
    faces = face_cascade.detectMultiScale(gray, 1.3, 7)     #détecte les visages, .detectMultiScale(image grisée, facteur d'échelle, nombre mini de voisin que chaque candidat rectangle devrait avoir pour le conserver
    eyes = eye_cascade.detectMultiScale(gray, 1.5, 7)           #Le 3e parametre c'est le nb de rectangles voisins, en gros l'algo va poser différents rectangles candidats, et + je mets le parametre haut alors + ce sera précis mais y'aura - de détections (nb de rectangles candidats qui doivent se chevaucher)
                                                        #Le facteur d'échelle sert à indiquer de combien réduire la fenetre de taille (qui va balayer l'écran) pour check les visages (1.5 -> Réduit de 50% à chaque itération)
    #.detectMultiScale() renvoie un rectangle (x,y,largeur,hauteur) qui est un oeil détecté
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)        #dessine un rectangle vert autour des yeux d'épaisseur 2

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (125, 125, 125), 2)
        roi_gray = gray[y:y + h, x:x + w]           #Roi: Région of interest, dans les gris et dans les couleurs. C'est les rectangles de visage détectés, et on vient extraire JUSTE les visages
        roi_color = img[y:y + h, x:x + w]

        mouths = mouth_cascade.detectMultiScale(roi_gray, 5, 5)     #On va détecter les bouches sur la fenetre réduite au visage
        for (sx, sy, sw, sh) in mouths:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

        noses = nose_cascade.detectMultiScale(roi_gray, 1.6, 5)     #On va détecter les nez sur la fenêtre réduite au visage
        for (nx, ny, nw, nh) in noses:
            cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 2)

    if len(eyes) != 0 and len(mouths) == 0 and len(noses) == 0: #Si il n'y a pas d'yeux ni bouches ni nez, alors ok+=1
        ok+=1   #ok est l'évaluateur positif (port du masque yes)
    c+=1 #compteur, nombre d'essai
    if c>=100:          #Au bout de 100 essais, fait une évaluation
        if ok>=0.5*c:   #Si y'a au moins 50% d'essais qui sont ok alors
            cv2.putText(img, "MASQUE OK", (ex, ey), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        else:   #sinon < 50 donc pas de masque
            cv2.putText(img, "PAS DE MASQUE", (ey, ex), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    if c>=200:  #à 200 ça vient reinitialiser
        c,ok=0,0
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    print(c,ok)

cap.release()
cv2.destroyAllWindows()

#Test de reussite : tourner 5 sec, voir si il y a masque porté plus de 70% du temps, si oui mettre ok pour 5 sec