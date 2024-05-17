import face_recognition
import numpy as np
import cv2
import os
from os import makedirs
import platform

imagenRostro="../photos_test/fotoRostroKevin.jpeg"
imagenIne="../photos_test/frenteINEKevin.jpeg"

def compare(**kwargs):
    
    ROOT_DIR = os.path.dirname(__file__)
    makedirs(f'{ROOT_DIR}/.configurationfiles', exist_ok=True)
    total=[]
    try:
        known_image=cv2.imread(kwargs['Know'])
        known_image = cv2.resize(known_image, (640,480), interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(known_image, cv2.COLOR_BGR2GRAY)
        
        
        if platform.system() == 'Windows':
            face_cascade = cv2.CascadeClassifier(f"../model_faces/haarcascade_frontalface_default.xml")
        elif platform.system() == 'Linux':
            face_cascade = cv2.CascadeClassifier(f"../model_faces/haarcascade_frontalface_default.xml")
        else:
            print("""
                THIS SCRIPT ONLY WORKINK WITH LINUX OR WINDOWS SYSTEM,
                PLEASE CONFIG FOR YOUR SYSTEM.
            """)
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) ==1:
            known_image = face_recognition.load_image_file(kwargs['Know'])
            known_encoding = face_recognition.face_encodings(known_image)[0]
    except:
        return "Si no definio la imagen base use el argumento 'Know = Ruta_de_la_imagen.extension', en caso contrario no se detecto el rostro en la imagen"
    else:
        try:
            image = cv2.imread(kwargs['Unknown'])
        except:
            return "No se encuentra la imagen a comparar, use el argumento Unknown = Ruta_de_la_imagen.extension"
        else:
            detection = False
            i=0
            while True:
                print(f"Comparing credencial {i}")
                if detection ==True or i >= 20:
                    break
                try:
                    rotation_angle= i * 20
                    print(rotation_angle)
                    (h, w) = image.shape[:2]
                    (cX, cY) = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D((cX, cY), rotation_angle, 1.0)
                    rotated = cv2.warpAffine(image, M, (w, h))
                    print("Rotando para encontrar coincidencia")
                    
                    
                    if platform.system() == 'Windows':
                        cv2.imwrite(f'{ROOT_DIR}\\.configurationfiles\\rotation.jpg', rotated)
                        print("Mejor posicion de la credencial")
                        Unknown_image = face_recognition.load_image_file(".configurationfiles\\rotation.jpg")
                    elif platform.system() == 'Linux':
                        cv2.imwrite(f'{ROOT_DIR}/.configurationfiles/rotation.jpg', rotated)
                        Unknown_image = face_recognition.load_image_file(".configurationfiles/rotation.jpg")
                    else:
                        print("""
                            THIS SCRIPT ONLY WORKINK WITH LINUX OR WINDOWS SYSTEM,
                            PLEASE CONFIG FOR YOUR SYSTEM.
                        """)
                    
                    unknown_encoding = face_recognition.face_encodings(Unknown_image)[0]
                    same_person = face_recognition.compare_faces([known_encoding], unknown_encoding)
                    
                except:
                    pass
                else:
                    if True in same_person:
                        detection=True
                        total.append(True)
                i+=1
            if detection == False:
                print ("No se reconocio la cara")
            else:
                print("Se reconocio el rostro en el Ine")
            try:
                video = kwargs['Video']
            except:
                if detection == True:
                    return f"Se ha encontrado al menos una coincidencia, si quiere realizar la comparacion contra un video agregue el argumento 'Video= ruta_del_video.extension'"
                else:
                    return f"No se encontro coincidencia, si quiere realizar la comparacion contra un video agregue el argumento 'Video= ruta_del_video.extension'"
            else:
                detection =False
                detected_faces_counter = 0
                try:
                    cap = cv2.VideoCapture(video)
                except:
                    return "No se puede acceder al video"
                else:
                    if not cap.isOpened():
                        return "El video tiene un error no se pudo leer, o no se encuentra el archivo"
                    detection = False
                    testing = 0
                    while True:
                        if detection == True:
                            break
                        ret, frame = cap.read()
                        if not ret:
                            break
                        else:
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                            if len(faces) >= 1:
                                i=0
                                while True:
                                    if i >= 20 or detection == True:
                                        break
                                    print(f"Comparando Video {i}")
                                    rotation_angle= i * 20
                                    (h, w) = frame.shape[:2]
                                    (cX, cY) = (w // 2, h // 2)
                                    M = cv2.getRotationMatrix2D((cX, cY), rotation_angle, 1.0)
                                    rotated = cv2.warpAffine(frame, M, (w, h))
                                    print("rotando")
                                    
                                    if platform.system() == 'Windows':
                                        cv2.imwrite(f'{ROOT_DIR}\\.configurationfiles\\video.jpg', rotated)
                                    elif platform.system() == 'Linux':
                                        cv2.imwrite(f'{ROOT_DIR}/.configurationfiles/video.jpg', rotated)
                                    else:
                                        print("""
                                            THIS SCRIPT ONLY WORKINK WITH LINUX OR WINDOWS SYSTEM,
                                            PLEASE CONFIG FOR YOUR SYSTEM.
                                        """)
                                    try:
                                        
                                        if platform.system() == 'Windows':
                                            Unknown_image = face_recognition.load_image_file(f'{ROOT_DIR}\\.configurationfiles\\video.jpg')
                                        elif platform.system() == 'Linux':
                                            Unknown_image = face_recognition.load_image_file(f'{ROOT_DIR}/.configurationfiles/video.jpg')
                                        else:
                                            print("""
                                                THIS SCRIPT ONLY WORKINK WITH LINUX OR WINDOWS SYSTEM,
                                                PLEASE CONFIG FOR YOUR SYSTEM.
                                            """)
                                        
                                        unknown_encoding = face_recognition.face_encodings(Unknown_image)[0]
                                        same_person = face_recognition.compare_faces([known_encoding], unknown_encoding)
                                        print(f"Deteccion : {same_person}")
                                        if True in same_person:
                                            detection=True
                                            print(f"Se encontro el rostro dentro del video")
                                            total.append(True)
                                    except:
                                        pass
                                    i+=1
                            testing +=1
                            print(f"{testing} imagenes han sido probadas")
                    cap.release()
                    return f"Promedio de deteccion: { (len(total)+1) /3}"



if __name__ == "__main__":
    print( compare(Know= imagenRostro, Unknown= imagenIne))