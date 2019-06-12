# Face detection on Raspberry Pi using OpenCV

Special thanks to [_Marcelo Rovai_](https://github.com/Mjrovai) for the code and the great [tutorial](https://www.hackster.io/mjrobot/real-time-face-recognition-an-end-to-end-project-a10826).

I have also used code from the [picamera docs](https://picamera.readthedocs.io/en/release-1.10/recipes1.html).

I have used the haarcascade_frontalface_default.xml from [OpenCV](https://github.com/opencv/opencv/tree/master/data/haarcascades).

---

1. Create your dataset with the faces of your different subjects.

2. Train the model.

3. Start the face recognition stream on the host, then on the Raspberry Pi - the client.

---

![alt text](https://github.com/ncsereoka/rpi_haarcascade/blob/master/image.jpg "Recognition")