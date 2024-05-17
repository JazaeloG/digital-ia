import cv2
import math
import argparse
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
from flask import Flask, request, jsonify
import numpy as np
import dlib 

app = Flask(__name__)


class faceNet():

    def __init__(self, path="/app/pesos-modelo/pre-trained_weights/modelo_facenet.h5"):

        self.model = load_model("/app/pesos-modelo/pre-trained_weights/modelo_facenet.h5", custom_objects={"Adamw": tfa.optimizers.AdamW}, compile=False)
        self.facialList = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
                           'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
                           'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
                           'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
                           'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
                           'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
                           'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
                           'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
                           'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
                           'Wearing_Necktie', 'Young']
        self.interest_classes = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Eyeglasses', 'Gray_Hair', 'Oval_Face', 'Pale_Skin', 'Straight_Hair', 'Wavy_Hair']
        self.detector = dlib.get_frontal_face_detector()

    def run(self, image, thresh=0.5):
        image_batch = np.zeros((1, 128, 128, 3))

        dets = self.detector(image)
        face_results = []
        for det in dets:
            faceTemp = {}
            results = []
            coord = [det.left(), det.top(), det.right(), det.bottom()]
            cropImage = image[det.top(): det.bottom(), det.left(): det.right()]
            image_batch[0] = cv2.resize(cropImage, (128, 128)) / 256
            output = self.model.predict(image_batch)

            for i, class_name in enumerate(self.facialList):
                if class_name in self.interest_classes:
                    temp = {}
                    temp["label"] = class_name
                    temp["prob"] = str(output[0][i])
                    results.append(temp)

            faceTemp["face"] = results
            faceTemp["coord"] = coord
            face_results.append(faceTemp)

        return face_results


@app.route('/', methods=["POST"])
def index():

    image = request.files['file'].read()
    npimg = np.frombuffer(image, np.uint8)
    image_np = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = faceAPP.run(image_np)

    result_json = {}
    result_json["result"] = results
    return jsonify(result_json)


if __name__ == '__main__':
    faceAPP = faceNet()
    app.run(debug=True, host='0.0.0.0')
