FROM tensorflow/tensorflow:2.8.0

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install opencv-python-headless
RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
RUN pip install flask
RUN pip install requests
RUN pip install numpy
RUN pip install tensorflow-addons
RUN pip install cmake
RUN pip install dlib

COPY ./pesos-modelo/pre-trained_weights/modelo_facenet.h5 /app/pesos-modelo/pre-trained_weights/modelo_facenet.h5

COPY AtributosFaciales.py /app/AtributosFaciales.py

EXPOSE 5000

CMD ["python", "/app/AtributosFaciales.py"]
