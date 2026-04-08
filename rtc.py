import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time
from googletrans import Translator
from gtts import gTTS


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
st.markdown("<h1>🖐️ ASL Hand Sign Detection & Translation</h1>", unsafe_allow_html=True)

# Session states
if "detected_sentence" not in st.session_state:
    st.session_state.detected_sentence = ""

if "last_detection_time" not in st.session_state:
    st.session_state.last_detection_time = time.time()

if "last_detected_letter" not in st.session_state:
    st.session_state.last_detected_letter = None

if "translated_sentence" not in st.session_state:
    st.session_state.translated_sentence = ""

translator = Translator()

language_dict = {
"English":"en","Hindi":"hi","Tamil":"ta",
"French":"fr","German":"de","Spanish":"es",
"Japanese":"ja","Korean":"ko"
}

# Buttons
col1, col2, col3 = st.columns(3)

with col1:
    start_camera = st.button("Start Camera")

with col2:
    stop_camera = st.button("Stop Camera")

with col3:
    clear_sentence = st.button("Clear Sentence")

# Clear
if clear_sentence:
    st.session_state.detected_sentence = ""
    st.session_state.translated_sentence = ""

# Display sentence
sentence_placeholder = st.empty()
sentence_placeholder.write(f"Detected Sentence: {st.session_state.detected_sentence}")

# Language selection
selected_language = st.selectbox("Select Language", list(language_dict.keys()))

# Translation
if st.button("Translate"):
    text = st.session_state.detected_sentence.strip()

    if text:
        translated = translator.translate(text, dest=language_dict[selected_language])
        st.session_state.translated_sentence = translated.text

        tts = gTTS(st.session_state.translated_sentence, lang=language_dict[selected_language])
        tts.save("translated_audio.mp3")

        st.audio("translated_audio.mp3")

translated_placeholder = st.empty()
translated_placeholder.write(f"Translated Text: {st.session_state.translated_sentence}")

# ---------------- CAMERA ---------------- #
cap = cv2.VideoCapture(0)

frame_window = st.image([])
frame_count = 0

while start_camera and not stop_camera:

    ret, frame = cap.read()

    if not ret:
        st.warning("Camera error")
        break

    frame = cv2.flip(frame, 1)

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    data_aux = []
    x_ = []
    y_ = []

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        for lm in hand_landmarks.landmark:
            x_.append(lm.x)
            y_.append(lm.y)

        for i in range(len(hand_landmarks.landmark)):
            data_aux.append(x_[i] - min(x_))
            data_aux.append(y_[i] - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        frame_count += 1

        if frame_count % 3 == 0:

            prediction = model.predict(np.asarray(data_aux[:42]).reshape(1, -1))
            predicted_character = prediction[0]

            if predicted_character != st.session_state.last_detected_letter:

                st.session_state.last_detection_time = time.time()
                st.session_state.last_detected_letter = predicted_character

            elif time.time() - st.session_state.last_detection_time > 1:

                if predicted_character == "del":
                    st.session_state.detected_sentence = st.session_state.detected_sentence[:-1]

                elif predicted_character == "space":
                    st.session_state.detected_sentence += " "

                elif predicted_character == "nothing":
                    pass

                else:
                    st.session_state.detected_sentence += predicted_character

                st.session_state.last_detection_time = time.time()

                sentence_placeholder.write(
                    f"Detected Sentence: {st.session_state.detected_sentence}"
                )

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

        cv2.putText(
            frame,
            str(st.session_state.last_detected_letter),
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

    frame_window.image(frame, channels="BGR")

cap.release()
cv2.destroyAllWindows()
