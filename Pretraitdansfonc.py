import cv2
import dlib
import numpy as np
import os
from scipy.spatial import distance
from PIL import Image
import mysql.connector
conn = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="root",
    database="donneespatient")

predictor_path = "C:/Users/dylan/OneDrive/Bureau/ProjetMask/eye_eyebrows_22.dat"
predictor = dlib.shape_predictor(predictor_path)

base_image_path = "C:\\Users\\dylan\\OneDrive\\Bureau\\ProjetMask\\Bdd\\with_mask\\"
output_path = "C:\\Users\\dylan\\OneDrive\\Bureau\\ProjetMask\\Bddppt3\\"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def calculate_ratio(value1, value2):
    if value2 != 0:
        return value1 / value2
    else:
        return 0

def calculate_distance(point1, point2):
    return distance.euclidean((point1.x, point1.y), (point2.x, point2.y))

def calculate_EAR(points):  
    A = distance.euclidean(points[1], points[5])
    B = distance.euclidean(points[2], points[4])
    C = distance.euclidean(points[0], points[3])
    EAR = (A + B) / (2.0 * C)
    return EAR

def calculate_EER(eye_points, eyebrow_points): 
    eye_length = distance.euclidean(eye_points[0], eye_points[3])
    eyebrow_length = distance.euclidean(eyebrow_points[0], eyebrow_points[4])
    EER = eyebrow_length / eye_length
    return EER


def process_imagesbdd():    #Cette fonction prends toute la base d'images, fous les points sur les photos, calcule les ratios et renvoie la photo + ses ratios
    ratios = []
    for k in range(500):
        # Construct the image path
        image_path = base_image_path + str(k) + "-with-mask.jpg"
        if os.path.isfile(image_path):
            img = cv2.imread(image_path)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
            roi_hsv = None
            for (x, y, w, h) in faces:
                roi_hsv = img_gray[y:y + h, x:x + w] 
                face = dlib.rectangle(x, y, x+w, y+h)
                shape = predictor(img, face)
                ratioint = []
                left_eyebrow_points= []
                right_eyebrow_points= []
                courbures = []
                if roi_hsv is not None:
                    for j in range(22): 
                        point = shape.part(j)
                        cv2.circle(img, (point.x, point.y), 1, (0, 0, 255), -1)
                    cv2.imwrite(output_path + str(k) + "-cleared.jpg", img[y:y + h, x:x + w])
                
                # Calculate ratios
                    interocular_distance = calculate_distance(shape.part(13), shape.part(16))
                    left_eye_width = calculate_distance(shape.part(10), shape.part(13))
                    right_eye_width = calculate_distance(shape.part(16), shape.part(19))
                    left_eyebrow_width = calculate_distance(shape.part(0), shape.part(4))
                    right_eyebrow_width = calculate_distance(shape.part(5), shape.part(9))
                    eye_height_to_eyebrow_height = []
                    for i in range(0,5):
                        k = shape.part(i).x
                        l = shape.part(i).y
                        left_eyebrow_points.append((k,l))
                    for i in range(5,10):
                        m = shape.part(i).x
                        p = shape.part(i).y
                        right_eyebrow_points.append((m,p))
                    distancesG = [distance.euclidean(left_eyebrow_points[i], left_eyebrow_points[i+1]) for i in range(len(left_eyebrow_points)-1)]
                    courbures.append(np.mean(distancesG))
                    distancesD = [distance.euclidean(right_eyebrow_points[i], right_eyebrow_points[i+1]) for i in range(len(right_eyebrow_points)-1)]
                    courbures.append(np.mean(distancesD))
                    for i in range(10, 16): 
                        eye_height = shape.part(i).y - shape.part(i-6).y
                        eyebrow_height = shape.part(i-6).y - min([shape.part(j).y for j in range(10, 16)])
                        eye_height_to_eyebrow_height.append(calculate_ratio(eyebrow_height, eye_height))

                    for i in range(16, 22):
                        eye_height = shape.part(i).y - shape.part(i-11).y
                        eyebrow_height = shape.part(i-11).y - min([shape.part(j).y for j in range(16, 22)])
                        eye_height_to_eyebrow_height.append(calculate_ratio(eyebrow_height, eye_height))

                    ratioint.extend([
                        calculate_ratio(left_eye_width, interocular_distance),
                        calculate_ratio(right_eye_width, interocular_distance),
                        calculate_ratio(left_eyebrow_width, interocular_distance),
                        calculate_ratio(right_eyebrow_width, interocular_distance)
                        ])
                    ratioint.extend(eye_height_to_eyebrow_height)
                    ratios.append((image_path, ratioint,courbures))
                else:
                    print("No face detected")
        else:
            print(f"Image {image_path} not found.")
    return ratios

#image_path = "C:\\Users\\dylan\\OneDrive\\Bureau\\ProjetMask\\Bdd\\test.jpg" 
def process_imagesolo(image_path):
    ratiosolo = []
    if os.path.isfile(image_path):
        img = cv2.imread(image_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # img_gray sert à la cascade de haar d'être + efficace dans la reconnaissance de visage
        faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
        roi_hsv = None
        for (x, y, w, h) in faces:
            roi_hsv = img_gray[y:y + h, x:x + w] 
            face = dlib.rectangle(x, y, x+w, y+h)
            shape = predictor(img, face)
            ratioint = []
            left_eyebrow_points= []
            right_eyebrow_points= []
            courbures = []
            
            if roi_hsv is not None:
                for j in range(22):  # There are 22 landmarks in total
                    point = shape.part(j)
                    cv2.circle(img, (point.x, point.y), 1, (0, 0, 255), -1) #je trace les points sur l'image de base, et je garde l'image de base pcq je veux les couleurs
                cv2.imwrite("C:\\Users\\dylan\\OneDrive\\Bureau\\ProjetMask\\Bdd\\testok.jpg" , img[y:y + h, x:x + w])
                interocular_distance = calculate_distance(shape.part(13), shape.part(16))
                left_eye_width = calculate_distance(shape.part(10), shape.part(13))
                right_eye_width = calculate_distance(shape.part(16), shape.part(19))
                left_eyebrow_width = calculate_distance(shape.part(0), shape.part(4))
                right_eyebrow_width = calculate_distance(shape.part(5), shape.part(9))
                eye_height_to_eyebrow_height = []
                for i in range(0,5):
                    k = shape.part(i).x
                    l = shape.part(i).y
                    left_eyebrow_points.append((k,l))
                for i in range(5,10):
                    p = shape.part(i).x
                    m = shape.part(i).y
                    right_eyebrow_points.append((p,m))
                distancesG = [distance.euclidean(left_eyebrow_points[i], left_eyebrow_points[i+1]) for i in range(len(left_eyebrow_points)-1)]
                courbures.append(np.mean(distancesG))
                distancesD = [distance.euclidean(right_eyebrow_points[i], right_eyebrow_points[i+1]) for i in range(len(right_eyebrow_points)-1)]
                courbures.append(np.mean(distancesD))
                    
                for i in range(10, 16): 
                    eye_height = shape.part(i).y - shape.part(i-6).y
                    eyebrow_height = shape.part(i-6).y - min([shape.part(j).y for j in range(10, 16)])
                    eye_height_to_eyebrow_height.append(calculate_ratio(eyebrow_height, eye_height))

                for i in range(16, 22):
                    eye_height = shape.part(i).y - shape.part(i-11).y
                    eyebrow_height = shape.part(i-11).y - min([shape.part(j).y for j in range(16, 22)])
                    eye_height_to_eyebrow_height.append(calculate_ratio(eyebrow_height, eye_height))

                ratioint.extend([
                calculate_ratio(left_eye_width, interocular_distance),
                calculate_ratio(right_eye_width, interocular_distance),
                calculate_ratio(left_eyebrow_width, interocular_distance),
                calculate_ratio(right_eyebrow_width, interocular_distance)])
                ratioint.extend(eye_height_to_eyebrow_height)
                ratiosolo.append((image_path, ratioint,courbures))
            else:
                print("No face detected")
    else:
        print(f"Image {image_path} not found.")
    return ratiosolo
        
cursor = conn.cursor()

def RemplissagePatientsbdd():
    ratios = process_imagesbdd()
    ratios_str = ','.join(str(ratio) for ratio in ratios)

    # Requête SQL d'insertion
    insert_query = "INSERT INTO Patient (Nom, Prenom, Ratios, FileVisage) VALUES (%s, %s, %s, %s)"
    Listenom = ["Sundar"+ str(i) for i in range(1000)]  # Remplacer par le nom approprié
    Listeprenom = ["Pichai"+str(i) for i in range(1000)]
    # Parcourir la liste des ratios
    k = 0
    for ratio_data in ratios:
        nom = Listenom[k] 
        prenom = Listeprenom[k]
        k+=1
        file_visage = ratio_data[0]
        ratios_str = ','.join(str(ratio) for ratio in ratio_data[1])
        
        # Valeurs pour l'insertion
        values = (nom, prenom, ratios_str, file_visage)

        # Exécuter la requête d'insertion
        cursor.execute(insert_query, values)

    # Valider la transaction
    conn.commit()
    cursor.close()
    conn.close()






def NouveauPatient(file_visage,ratios):
    print("Veuillez entrez le nom du patient correspondant")
    nom = str(input())
    print("Veuillez entrez le prénom du patient correspondant")
    prenom = str(input())

    ratios_str = ','.join(str(ratio) for ratio in ratios)

# Requête SQL d'insertion
    insert_query = "INSERT INTO Patient (Nom, Prenom, Ratios, FileVisage) VALUES (%s, %s, %s, %s)"
    values = (nom, prenom, ratios_str, file_visage)

# Exécuter la requête d'insertion
    cursor.execute(insert_query, values)

# Valider la transaction
    conn.commit()
    cursor.close()
    conn.close()
    



def compare_ecarts(ratio1tot, image1, ratio2, image2):
    Lecarts_relatifs = []
    for index, sous_liste in enumerate(ratio1tot):
        sous_liste = np.array(sous_liste)  # Convertir la sous-liste en un tableau NumPy
        ecart_relatif = np.abs((sous_liste - ratio2) / ratio2) * 100
        ecart_relatif = ecart_relatif.mean()  # Calculer la moyenne des écarts relatifs
        Lecarts_relatifs.append((ecart_relatif, index))
    Lmeilleures_sous_listes = []
    for _ in range(10):
        Lecarts_relatifs.sort(key=lambda x: x[0])
        meilleure_sous_liste = ratio1tot[Lecarts_relatifs[0][1]]
        Lmeilleures_sous_listes.append((meilleure_sous_liste, Lecarts_relatifs[0][1]))
        Lecarts_relatifs.pop(0)
    for sous_liste, indice in Lmeilleures_sous_listes:
        print(indice)

        #Je veux comparer couleur des veuch, des yeux et des sourcils, si c'est ok c'est bien l'individu


def compare_courbes(courbdd, courbsolo):
    distances = []
    for index, sublist in enumerate(courbdd):
        sublist = np.array(sublist)
        ecart_relatif = np.abs((sublist - courbsolo) / courbsolo) * 100
        ecart_relatif = ecart_relatif.mean()
        distances.append((ecart_relatif, index))
    distances.sort()  # Sort the distances in ascending order
    
    # Retrieve the indices of the top 10 closest sublists
    top_20_indices = [index for _, index in distances[:20]]
    return top_20_indices
    
    return top_10_indices



def compare_pixels(image1_path, image2_path, x, y, tolerance):
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    pixel1 = image1.getpixel((x, y))
    pixel2 = image2.getpixel((x, y))

    for i in range(3):  # R, G, B channels
        diff = abs(pixel1[i] - pixel2[i])
        max_diff = max(pixel1[i], pixel2[i]) * tolerance

        if diff > max_diff:
            return False

    return True

def comparaison():
    path_imagesolo = "C:\\Users\\dylan\\OneDrive\\Bureau\\ProjetMask\\Bdd\\484-with-mask.jpg"
    resultatbdd = process_imagesbdd()
    imagesbdd, ratiobdd,courbdd = zip(*resultatbdd)
    resultatsolo = process_imagesolo(path_imagesolo)
    imagesolo, ratiosolo,courbsolo = zip(*resultatsolo)
    top_20_courbes_i = compare_courbes(courbdd,courbsolo)
    triratiobdd = []
    for i in top_20_courbes_i:
        triratiobdd += ratiobdd[i]
    compare_ecarts(triratiobdd,imagesbdd,ratiosolo,imagesolo)
    