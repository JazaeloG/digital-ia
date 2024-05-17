import os
from sklearn.model_selection  import train_test_split
import numpy as np
import argparse
from model_facenet import faceNet
from model_vit import vitNet


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description='facial-attribute-extraction')
    parser.add_argument("--imagepath", type=str,dest="data_path" ,help="Path hacia las imagenes ",default='./imagenes.npz',action="store")
    parser.add_argument("--labelpath", type=str,dest="label_path" ,help="Path hacia las etiquetas ",default='./nuevasLabels.npz',action="store")
    parser.add_argument("--model", type=str,dest="model_type" ,help="Modelo a usar para entrenamiento",default='facenet',action="store")
    
    args = parser.parse_args()
      
    assert args.data_path[-3:]=="npz","El tipo de archivo de entrenamiento debe ser npz. Reemplace el archivo de entrenamiento"
    assert args.label_path[-3:]=="npz","El tipo de archivo de entrenamiento debe ser npz. Reemplace el archivo de entrenamiento"
    
    
    ## cargar los datos de entrenamiento
    
    data_x = np.load( args.data_path, allow_pickle=True)
    data_y = np.load(args.label_path, allow_pickle=True)
    data_x = data_x['imagenes']
    data_y = data_y['labels']
    
    # convertir las etiquetas a 0 y 1
    data_y[data_y==-1] = 0
    
    # limitar el numero de datos de entrenamiento
    data_x=data_x[0:50000]
    data_y=data_y[0:50000]
    
    
    # separar datos de entrenamiento con datos de prueba
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)

    
    del data_x,data_y # eliminar los datos de entrenamiento para liberar memoria
    
    # normalizacion de los datos
    
    x_train = np.float32(x_train/255)
    
    x_test = np.float32(x_test/255)
    
    labels = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Eyeglasses', 'Gray_Hair', 'Oval_Face', 'Pale_Skin', 'Straight_Hair', 'Wavy_Hair']
    
    
    # Preocesado de los datos de entrenamiento
    if args.model_type=="facenet":
        print("facenet model training started...")

        model = faceNet(img_width=128,img_height=128) 
        model.build()
        model.run(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,validation_split=0.1)
        model.save("modelo_entrenado.h5")
    elif args.model_type=="vit":
        print("vit classifier model training started...")

        model = vitNet() 
        model.create_vit_classifier(input_shape = (128, 128, 3),num_classes = 40)
        model.run(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,validation_split=0.1)
        model.save("modelo_entrenado.h5")
        
        
    print("Training has been completed")