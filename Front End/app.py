from flask import Flask, render_template, Response, jsonify
import cv2 as cv
import mediapipe as mp
import numpy as np
import pickle
import time
import pyttsx3

app = Flask(__name__)

# Load the ASL model from the pickle file
model_asl_dict = pickle.load(open('model_ASL1.pkl', 'rb'))
model_asl = model_asl_dict['model']

# Initialize MediaPipe hands for hand landmark detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Define gesture labels for ASL
labels_dict = {
    "A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "F": "F", "G": "G", "H": "H", "I": "I",
    "J": "J", "K": "K", "L": "L", "M": "M", "N": "N", "O": "O", "P": "P", "Q": "Q", "R": "R",
    "S": "S", "T": "T", "U": "U", "V": "V", "W": "W", "X": "X", "Y": "Y", "Z": "Z",
    "space": " ", "nothing": ""  # Adding support for space and no gesture
}

predicted_text = ""  # Initialize variable to hold the predicted text


# Route for the main menu
@app.route('/')
def main_menu():
    return render_template('main.html')


# Route to set up the ASL recognition page
@app.route('/asl')
def asl_page():
    return render_template('index.html', language="AMERICAN SIGN LANGUAGE")


# Route to serve the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Function to generate frames from the webcam
def generate_frames():
    global predicted_text
    cap = cv.VideoCapture(0)  # Open webcam
    prev_sign = None  # Store the previously detected sign
    start_time = None  # Store the start time of detection
    detection_threshold = 1  # Time in seconds to confirm detection

    while True:
        data_aux = []  # To store landmark coordinates
        x_ = []  # To store x coordinates of landmarks
        y_ = []  # To store y coordinates of landmarks
        ret, frame = cap.read()
        if not ret:
            break
        H, W, _ = frame.shape
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)
            x1 = int(min(x_) * W)
            y1 = int(min(y_) * H)
            x2 = int(max(x_) * W)
            y2 = int(max(y_) * H)

            # Ensure data_aux has the right length for the model
            if len(data_aux) == 42:
                data_aux.extend([0] * 42)
            elif len(data_aux) > 84:
                data_aux = data_aux[:84]

            # Predict the character using the ASL model
            prediction = model_asl.predict([np.asarray(data_aux)])
            predicted_character = labels_dict.get(str(prediction[0]), "Unknown")

            # Draw a rectangle around the hand and put the predicted character on the frame
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), thickness=4)
            cv.putText(frame, predicted_character, (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv.LINE_AA)

            # If the same character is detected for a continuous period, add it to the predicted text
            if predicted_character == prev_sign:
                if start_time is None:
                    start_time = time.time()
                elif time.time() - start_time >= detection_threshold:
                    predicted_text += predicted_character
                    start_time = None  # Reset timer for next detection
            else:
                prev_sign = predicted_character
                start_time = None

        ret, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv.destroyAllWindows()


# Route to clear the last character in the predicted text
@app.route('/clear_last_character', methods=['POST'])
def clear_last_character():
    global predicted_text
    if predicted_text:
        predicted_text = predicted_text[:-1]
    return jsonify(predicted_text=predicted_text)


# Route to speak out the predicted sentence using text-to-speech
@app.route('/speak_sentence', methods=['POST'])
def speak_sentence():
    global predicted_text
    engine = pyttsx3.init()
    engine.say(predicted_text)
    engine.runAndWait()
    return '', 204


# Route to clear the entire predicted sentence
@app.route('/clear_sentence', methods=['POST'])
def clear_sentence():
    global predicted_text
    predicted_text = ""
    return jsonify(success=True)


# Route to get the current predicted text
@app.route('/get_predicted_text', methods=['GET'])
def get_predicted_text():
    return jsonify(predicted_text=predicted_text)


# Route to add a space in the predicted text
@app.route('/add_space', methods=['POST'])
def add_space():
    global predicted_text
    predicted_text += " "
    return jsonify(predicted_text=predicted_text)


if __name__ == '__main__':
    app.run(debug=True)
