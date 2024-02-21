from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import os
from typing import List, Dict
from dotenv import load_dotenv
from media_utils import (
    annotate_emotion_stats,
    annotate_warning,
    draw_bounding_box_annotation,
    draw_emoji,
    get_video_writer,
)
load_dotenv()
PATH = os.getenv("PROJECT_PATH")

class EmotionAnalysisVideo:
    """Class with methods to do emotion analysis on video or webcam feed."""

    # emoji_foldername = "emojis"
    def __init__(self, emoji_path, face_classifier, emotion_classifier, emotion_labels):

        # construct the path to emoji folder
        self.emoji_path = emoji_path
        # Load the emojis
        self.emojis = self.load_emojis(emoji_path=self.emoji_path)

        # Load the face detector
        self.face_detector = face_classifier
        self.emotion_detector = emotion_classifier
        self.emotion_labels = emotion_labels

    def emotion_analysis_video(
        self,
        video_path: str = None,
        save_output: bool = False,
        preview: bool = False,
        output_path: str = "data/output.mp4",
        resize_scale: float = 0.5,
        ) -> None:

        # if video_path is None:
        # If no video source is given, try
        # switching to webcam
        video_path = 0 if video_path is None else video_path

        cap, video_writer = None, None
        try:
            cap = cv2.VideoCapture(video_path)
            # To save the video file, get the opencv video writer
            video_writer = get_video_writer(cap, output_path)

            while True:
                    
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    # Flip webcam feed so that it looks mirrored
                    if video_path == 0:
                        frame = cv2.flip(frame, 2)
                except Exception as exc:
                    raise exc

                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(gray)

                for (x,y,w,h) in faces:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
                    roi_gray = gray[y:y+h,x:x+w]
                    roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

                    if np.sum([roi_gray])!=0:
                        roi = roi_gray.astype('float')/255.0
                        roi = img_to_array(roi)
                        roi = np.expand_dims(roi,axis=0)

                        prediction = self.emotion_detector.predict(roi)[0]
                        label = emotion_labels[prediction.argmax()]
                        label_position = (x,y)
                        cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

                        # Visulaize the emoji
                        # Annotate the current frame with emotion detection data
                        frame = self.annotate_emotion_data(label, frame, resize_scale)

                        if save_output:
                            video_writer.write(frame)
                        if preview:
                            cv2.imshow("Preview", cv2.resize(frame, (680, 480)))

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
        except Exception as exc:
            raise exc
        finally:
            cv2.destroyAllWindows()
            cap.release()
            video_writer.release()

    def load_emojis(self, emoji_path):
        emojis = {}

        # list of given emotions
        EMOTIONS = [
            "Angry",
            "Disgusted",
            "Fearful",
            "Happy",
            "Sad",
            "Surprised",
            "Neutral",
        ]

        # store the emoji coreesponding to different emotions
        for _, emotion in enumerate(EMOTIONS):
            emoji_path = os.path.join(self.emoji_path, emotion.lower() + ".png")
            emojis[emotion] = cv2.imread(emoji_path, -1)

        return emojis


    def annotate_emotion_data(
        self, emotion_data, image, resize_scale: float
    ):

        WARNING_TEXT = "Warning ! More than one person detected !"

        if len(emotion_data) > 1:
            image = annotate_warning(WARNING_TEXT, image)

        if len(emotion_data) > 0:
            image = draw_emoji(self.emojis[emotion_data], image)
            
        return image


if __name__ == "__main__":
    # classify_path = PATH + 'haarcascade_frontalface_default.xml'
    face_classifier = cv2.CascadeClassifier(PATH + 'haarcascade_frontalface_default.xml') # method is then used to detect faces in the input image, and rectangles are drawn around the detected faces.classifier =load_model(r'C:\Users\Admin\Documents\myProject\Facical-Recognition\Emotion_Detection_CNN\model.h5')
    emotion_classifier = load_model(PATH +'model.h5', compile=False)
    emotion_labels = ["Angry","Disgusted","Fearful","Happy","Sad","Surprised","Neutral"]
    emoji_path = './emojis/'

    emotion_recognizer = EmotionAnalysisVideo(
                            emoji_path=emoji_path,
                            face_classifier=face_classifier,
                            emotion_classifier=emotion_classifier,
                            emotion_labels=emotion_labels,
                        )

    emotion_recognizer.emotion_analysis_video(
        video_path=None,
        save_output=False,
        preview=True,
        output_path="data/output.mp4",
        resize_scale=0.5,
    )

