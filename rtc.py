import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time
from googletrans import Translator
from gtts import gTTS

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# ---------------- MODEL ---------------- #
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# ---------------- MEDIAPIPE ---------------- #
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------- LABELS ---------------- #
labels_dict = {
    0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',
    10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',
    18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z',
    26:'del',27:'nothing',28:'space'
}

# ---------------- UI ---------------- #
st.title("🖐️ ASL Hand Sign Detection & Translation")

if "sentence" not in st.session_state:
    st.session_state.sentence = ""

if "last_letter" not in st.session_state:
    st.session_state.last_letter = ""

if "last_time" not in st.session_state:
    st.session_state.last_time = time.time()

translator = Translator()

language_dict = {
    "English":"en","Hindi":"hi","Tamil":"ta",
    "French":"fr","German":"de","Spanish":"es",
    "Japanese":"ja","Korean":"ko"
}

# ---------------- CAMERA CLASS ---------------- #
class ASLTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        H, W, _ = img.shape
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            mp_drawing.draw_landmarks(
                img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_, y_, data_aux = [], [], []

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(x_[i] - min(x_))
                data_aux.append(y_[i] - min(y_))

            if len(data_aux) >= 42:
                prediction = model.predict(
                    np.asarray(data_aux[:42]).reshape(1, -1))
                predicted_character = prediction[0]

                # Sentence logic
                if predicted_character != st.session_state.last_letter:
                    st.session_state.last_time = time.time()
                    st.session_state.last_letter = predicted_character

                elif time.time() - st.session_state.last_time > 1:

                    if predicted_character == "del":
                        st.session_state.sentence = st.session_state.sentence[:-1]

                    elif predicted_character == "space":
                        st.session_state.sentence += " "

                    elif predicted_character != "nothing":
                        st.session_state.sentence += predicted_character

                    st.session_state.last_time = time.time()

                cv2.putText(
                    img,
                    str(predicted_character),
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

        return img

# ---------------- BUTTONS ---------------- #
col1, col2 = st.columns(2)

with col1:
    start = st.button("Start Camera")

with col2:
    clear = st.button("Clear Sentence")

if clear:
    st.session_state.sentence = ""

# ---------------- CAMERA ---------------- #
if start:
    webrtc_streamer(
        key="asl",
        video_transformer_factory=ASLTransformer
    )

# ---------------- TEXT DISPLAY ---------------- #
st.subheader("Detected Sentence:")
st.write(st.session_state.sentence)

# ---------------- TRANSLATION ---------------- #
selected_language = st.selectbox(
    "Select Language", list(language_dict.keys()))

if st.button("Translate"):
    text = st.session_state.sentence.strip()

    if text:
        translated = translator.translate(
            text, dest=language_dict[selected_language])

        st.subheader("Translated Text:")
        st.write(translated.text)

        tts = gTTS(translated.text, lang=language_dict[selected_language])
        tts.save("audio.mp3")

        st.audio("audio.mp3")
