import cv2
import numpy as np
import socket
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX


class RCDriverNNOnly(object):

    def __init__(self, host, port, model_path):

        self.server_socket = socket.socket()
        self.server_socket.bind((host, port))
        self.server_socket.listen(0)

        # accept a single connection
        self.connection = self.server_socket.accept()[0].makefile('rb')

        # load trained neural network
        #self.nn = NeuralNetwork()
        # self.nn.load_model(model_path)

    def drive(self):
        # iniciate id counter
        id = 0
        # names related to ids: Lizzie = 1
        names = ['None', 'Lizzie', 'Georgie']
        # Initialize and start realtime video capture

        minW = 64
        minH = 48

        stream_bytes = b' '
        try:
            # stream video frames one by one
            while True:
                stream_bytes += self.connection.read(1024)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')

                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    gray = cv2.imdecode(np.frombuffer(
                        jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    img = cv2.imdecode(np.frombuffer(
                        jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                    faces = faceCascade.detectMultiScale(
                        gray,
                        scaleFactor=1.2,
                        minNeighbors=5,
                        minSize=(int(minW), int(minH)),
                    )
                    for(x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                        # Check if confidence is less them 100 ==> "0" is perfect match
                        if (confidence < 100):
                            id = names[id]
                            confidence = "  {0}%".format(
                                round(100 - confidence))
                        else:
                            id = "unknown"
                            confidence = "  {0}%".format(
                                round(100 - confidence))

                        cv2.putText(img, str(id), (x+5, y-5),
                                    font, 1, (255, 255, 255), 2)
                        cv2.putText(img, str(confidence),
                                    (x+5, y+h-5), font, 1, (255, 255, 0), 1)

                    cv2.imshow('camera', img)
                    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
                    if k == 27:
                        break

        finally:
            print("\n [INFO] Exiting Program and cleanup stuff")
            cv2.destroyAllWindows()
            self.connection.close()
            self.server_socket.close()


if __name__ == '__main__':
    # host, port
    h, p = "0.0.0.0", 8000

    # model path
    path = "saved_model/nn_model.xml"

    rc = RCDriverNNOnly(h, p, path)
    rc.drive()
