import cv2
import numpy as np
import socket
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#from model import NeuralNetwork
#from rc_driver_helper import RCControl

face_id = input('\n enter user id end press <return> ==>  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count


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
        count = 0
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

                    faces = face_detector.detectMultiScale(gray, 1.3, 5)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        count += 1
                        # Save the captured image into the datasets folder
                        cv2.imwrite("dataset/User." + str(face_id) +
                                    '.' + str(count) + ".jpg", gray[y:y+h, x:x+w])
                        cv2.imshow('image', img)
                    # Press 'ESC' for exiting video
                    k = cv2.waitKey(100) & 0xff
                    if k == 27:
                        break
                    elif count >= 30:  # Take 30 face sample and stop video
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
