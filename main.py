import cv2, dlib, os, numpy as np
from imutils import face_utils, resize
from PIL import ImageTk
from PIL.Image import *
from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt


def visualizer():
    global cap
    global current_image
    if cap is not None:
        ret, frame = cap.read()
        if ret:
            faces = detector(frame)

            result = base_img.copy()

            if len(faces) > 0:
                face = faces[0]

                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                face_img = frame[y1:y2, x1:x2].copy()

                shape = predictor(frame, face)
                shape = face_utils.shape_to_np(shape)

                for p in shape:
                    cv2.circle(
                        face_img,
                        center=(p[0] - x1, p[1] - y1),
                        radius=2,
                        color=255,
                        thickness=-1,
                    )

                # eyes
                le_x1 = shape[36, 0]
                le_y1 = shape[37, 1]
                le_x2 = shape[39, 0]
                le_y2 = shape[41, 1]
                le_margin = int((le_x2 - le_x1) * 0.18)

                re_x1 = shape[42, 0]
                re_y1 = shape[43, 1]
                re_x2 = shape[45, 0]
                re_y2 = shape[47, 1]
                re_margin = int((re_x2 - re_x1) * 0.18)

                left_eye_img = frame[
                    le_y1 - le_margin : le_y2 + le_margin,
                    le_x1 - le_margin : le_x2 + le_margin,
                ].copy()
                right_eye_img = frame[
                    re_y1 - re_margin : re_y2 + re_margin,
                    re_x1 - re_margin : re_x2 + re_margin,
                ].copy()

                left_eye_img = resize(left_eye_img, width=130)
                right_eye_img = resize(right_eye_img, width=130)

                result = cv2.seamlessClone(
                    left_eye_img,
                    result,
                    np.full(left_eye_img.shape[:2], 255, left_eye_img.dtype),
                    (180, 190),
                    cv2.MIXED_CLONE,
                )

                result = cv2.seamlessClone(
                    right_eye_img,
                    result,
                    np.full(right_eye_img.shape[:2], 255, right_eye_img.dtype),
                    (320, 190),
                    cv2.MIXED_CLONE,
                )

                # mouth
                mouth_x1 = shape[48, 0]
                mouth_y1 = shape[50, 1]
                mouth_x2 = shape[54, 0]
                mouth_y2 = shape[57, 1]
                mouth_margin = int((mouth_x2 - mouth_x1) * 0.1)

                mouth_img = frame[
                    mouth_y1 - mouth_margin : mouth_y2 + mouth_margin,
                    mouth_x1 - mouth_margin : mouth_x2 + mouth_margin,
                ].copy()

                mouth_img = resize(mouth_img, width=250)

                result = cv2.seamlessClone(
                    mouth_img,
                    result,
                    np.full(mouth_img.shape[:2], 255, mouth_img.dtype),
                    (250, 320),
                    cv2.MIXED_CLONE,
                )

            frame = resize(result, width=500)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            current_image = fromarray(frame)
            image = ImageTk.PhotoImage(image=current_image)

            video.configure(image=image)
            video.image = image
            video.after(10, visualizer)


def initializer():
    global cap
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    visualizer()


def download_image():
    global i
    global current_image
    if current_image is not None:
        file_path = os.path.expanduser(f"~/Downloads/captured_image{i}.jpg")
        current_image.save(file_path)
        os.startfile(file_path)
        i += 1


def open_image_file():
    global base_img
    file_path = filedialog.askopenfilename()
    print(file_path)
    base_img = plt.imread(file_path)
    base_img = cv2.resize(base_img, dsize=(500, 500))
    base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)


i = 0
cap = None
current_image = None
base_img = cv2.imread("orange.jpg")
base_img = cv2.resize(base_img, dsize=(500, 500))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


root = Tk()

root.title("으하하 얼굴합성")
root.geometry("500x570")
root.resizable(width=False, height=False)

video = Label(root)
video.pack()

upload_button = Button(
    root, text="사진 파일 선택", width=12, height=2, command=open_image_file
)
upload_button.pack(padx=15, side=LEFT)

download_button = Button(root, text="캡처", width=12, height=2, command=download_image)
download_button.pack(padx=15, side=RIGHT)

initializer()

root.mainloop()
