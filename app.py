import sqlite3
import openai
import cv2
import mediapipe as mp
import numpy as np
import base64
import tensorflow as tf
import sounddevice as sd
import joblib
import pickle
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_socketio import SocketIO, emit
from werkzeug.security import generate_password_hash, check_password_hash

from config import OPENAI_API_KEY
from classification import preprocess_audio, predict_chord

app = Flask(__name__)
app.secret_key = '1234'
socketio = SocketIO(app)
DATABASE = 'instance/users.db'

openai.api_key = OPENAI_API_KEY

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load scaler, model and label encoder
model = tf.keras.models.load_model('model.h5')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

sample_rate = 22050
duration = 3

# Last feedback request timestamp
last_feedback_time = 0
feedback_interval = 5  # seconds

last_finger_positions = {
    'C': {'index': False, 'middle': False, 'ring': False},
    'D': {'index': False, 'middle': False, 'ring': False},
    'E': {'index': False, 'middle': False, 'ring': False},
    'G': {'index': False, 'middle': False, 'ring': False},
    'A': {'index': False, 'middle': False, 'ring': False}
}

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with app.app_context():
        db = get_db()
        with app.open_resource('instance/schema.sql', mode='r') as f:
            db.executescript(f.read())
        db.commit()

@app.route('/')
def index():
    username = session.get('username')
    return render_template('main.html', username=username)

@app.route('/about')
def about():
    username = session.get('username')
    return render_template('about.html', username=username)

@app.route('/tutorial_1')
def tutorial_1():
    username = session.get('username')
    return render_template('tutorial_1.html', username=username)

@app.route('/tutorial_2')
def tutorial_2():
    username = session.get('username')
    return render_template('tutorial_2.html', username=username)

@app.route('/basic_1')
def basic_1():
    username = session.get('username')
    return render_template('basic_1.html', username=username)

@app.route('/basic_2')
def basic_2():
    username = session.get('username')
    return render_template('basic_2.html', username=username)

@app.route('/basic_3')
def basic_3():
    username = session.get('username')
    return render_template('basic_3.html', username=username)

@app.route('/busking_1')
def busking_1():
    username = session.get('username')
    return render_template('busking_1.html', username=username)

@app.route('/busking_2')
def busking_2():
    username = session.get('username')
    return render_template('busking_2.html', username=username)

def save_record(username, song_title, mistakes):
    db = get_db()
    db.execute('INSERT INTO records (username, song_title, mistakes) VALUES (?, ?, ?)', (username, song_title, mistakes))
    db.commit()

def get_records(username):
    db = get_db()
    records = db.execute('SELECT song_title, mistakes FROM records WHERE username = ?', (username,)).fetchall()
    return records

@app.route('/records')
def records():
    username = session.get('username')
    if not username:
        return redirect(url_for('login'))
    records = get_records(username)
    return render_template('records.html', records=records)

@app.route('/save_record', methods=['POST'])
def save_record_route():
    data = request.get_json()
    username = data['username']
    song_title = data['song_title']
    mistakes = data['mistakes']
    save_record(username, song_title, mistakes)
    return jsonify({'success': True})

def is_valid_quad(points):
    if len(points) != 4:
        return False
    line_func = np.polyfit(points[:, 0], points[:, 1], 1, full=True)
    residuals = line_func[1]
    if len(residuals) == 0 or residuals[0] < 1e-10:
        return False
    return True

def calculate_distance(pt1, pt2):
    return np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)

def overlay_transparent(background, overlay, x_offset, y_offset):
    overlay_height, overlay_width = overlay.shape[:2]
    bg = background[y_offset:y_offset + overlay_height, x_offset:x_offset + overlay_width]
    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha
        for c in range(0, 3):
            bg[:, :, c] = (alpha * overlay[:, :, c] + alpha_inv * bg[:, :, c])
    else:
        bg = overlay
    background[y_offset:y_offset + overlay_height, x_offset:x_offset + overlay_width] = bg
    return background

# Feedback
def generate_feedback(chord, finger_positions):
    feedback = ""
    if chord == 'C':
        if not finger_positions['index']:
            feedback += "검지 손가락을 1프렛의 B줄에 위치시켜주세요. "
        if not finger_positions['middle']:
            feedback += "중지 손가락을 2프렛의 D줄에 위치시켜주세요. "
        if not finger_positions['ring']:
            feedback += "약지 손가락을 3프렛의 A줄에 위치시켜주세요. "

    elif chord == 'D':
        if not finger_positions['index']:
            feedback += "검지 손가락을 2프렛의 E줄에 위치시켜주세요. "
        if not finger_positions['middle']:
            feedback += "중지 손가락을 3프렛의 B줄에 위치시켜주세요. "
        if not finger_positions['ring']:
            feedback += "약지 손가락을 2프렛의 G줄에 위치시켜주세요. "
    
    elif chord == 'E':
        if not finger_positions['index']:
            feedback += "검지 손가락을 1프렛의 G줄에 위치시켜주세요. "
        if not finger_positions['middle']:
            feedback += "중지 손가락을 2프렛의 A줄에 위치시켜주세요. "
        if not finger_positions['ring']:
            feedback += "약지 손가락을 2프렛의 D줄에 위치시켜주세요. "   

    elif chord == 'G':
        if not finger_positions['index']:
            feedback += "검지 손가락을 3프렛의 E줄에 위치시켜주세요. "
        if not finger_positions['middle']:
            feedback += "중지 손가락을 2프렛의 A줄에 위치시켜주세요. "
        if not finger_positions['ring']:
            feedback += "약지 손가락을 3프렛의 B줄에 위치시켜주세요. "

    elif chord == 'A':
        if not finger_positions['index']:
            feedback += "검지 손가락을 2프렛의 D줄에 위치시켜주세요. "
        if not finger_positions['middle']:
            feedback += "중지 손가락을 2프렛의 G줄에 위치시켜주세요. "
        if not finger_positions['ring']:
            feedback += "약지 손가락을 2프렛의 B줄에 위치시켜주세요. "

    if feedback == "":
        feedback = "좋아요! 코드를 올바르게 잡고 있습니다."

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"다음 피드백을 자연스럽게 문장으로 만들어줘: {feedback}"}
        ],
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )

    advice = response.choices[0].message['content'].strip()
    return advice

@socketio.on('image')
def handle_image(data):
    global last_finger_positions

    sbuf = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
    frame = cv2.imdecode(sbuf, cv2.IMREAD_COLOR)

    frame = cv2.flip(frame, 1)
    
    # MediaPipe 처리
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # 손 랜드마크 인식 결과 처리
    hand_detected = False
    hand_inside_box = False

    if results.multi_hand_landmarks:
        hand_detected = True
        for hand_landmarks in results.multi_hand_landmarks:
            for finger_tip_id in [8, 12, 16, 20]:
                tip = hand_landmarks.landmark[finger_tip_id]
                # 손가락 끝 위치를 화면에 출력
                cv2.circle(frame, (int(tip.x * frame.shape[1]), int(tip.y * frame.shape[0])), 5, (0, 255, 0), -1)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue_lower = (100, 90, 80)
    blue_upper = (110, 255, 255)
    blue_mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_x, min_y = frame.shape[1], frame.shape[0]
    max_x, max_y = 0, 0

    if contours:
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
        all_points = []
        for contour in sorted_contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            all_points.extend(box)

        if all_points:
            # Calculate the minimum area rectangle that encloses all points
            all_points = np.array(all_points)
            min_rect = cv2.minAreaRect(all_points)
            center, (width, height), angle = min_rect

            # 높이를 20% 줄임
            new_height = height * 0.45

            # 새로운 사각형 정의
            new_rect = (center, (width, new_height), angle)
            big_box = cv2.boxPoints(new_rect)
            big_box = np.intp(big_box)
            cv2.polylines(frame, [big_box], isClosed=True, color=(255, 0, 0), thickness=2)

            # Apply Homography
            srcQuad = np.float32([big_box[1], big_box[2], big_box[3], big_box[0]])
            dstQuad = np.float32([[0, 0], [frame.shape[1], 0], [frame.shape[1], frame.shape[0]], [0, frame.shape[0]]])
            perspective_transform = cv2.getPerspectiveTransform(srcQuad, dstQuad)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for id in [8, 12, 16, 20]:  # 손끝 랜드마크 ID
                        landmark = hand_landmarks.landmark[id]
                        lx, ly = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                        if big_box[0][0] <= lx <= big_box[2][0] and big_box[0][1] <= ly <= big_box[2][1]:
                            hand_inside_box = True
                            break

    condition_c_index = False
    condition_c_middle = False
    condition_c_ring = False

    condition_d_index = False
    condition_d_middle = False
    condition_d_ring = False

    condition_e_index = False
    condition_e_middle = False
    condition_e_ring = False

    condition_g_index = False
    condition_g_middle = False
    condition_g_ring = False

    condition_a_index = False
    condition_a_middle = False
    condition_a_ring = False
    bounding_box = (min_x, min_y, max_x, max_y)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, landmark in enumerate(hand_landmarks.landmark):
                if id in [8, 12, 16, 20]:  # 손끝 랜드마크 ID
                    lx, ly = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    transformed_point = np.dot(perspective_transform, np.array([lx, ly, 1]))
                    transformed_point /= transformed_point[2]
                    tx, ty = int(transformed_point[0]), int(transformed_point[1])

                    # 바운딩 박스 내의 상대적 위치 계산 (바운딩 박스의 왼쪽 하단을 기준으로)
                    relative_x = abs(tx - bounding_box[0])
                    relative_y = abs(ty - bounding_box[1])

                    # 정규화
                    normalized_relative_x = relative_x / frame.shape[1]
                    normalized_relative_y = relative_y / frame.shape[0]

                    cv2.putText(frame, f'Relative: ({normalized_relative_x:.2f}, {normalized_relative_y:.2f})', (lx + 10, ly),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # C코드
                    if id == 8 and 0.7 <= normalized_relative_x <= 0.8 and 0.25 <= normalized_relative_y <= 0.40:
                        condition_c_index = True
                    elif id == 12 and 0.5 <= normalized_relative_x <= 0.6 and 0.55 <= normalized_relative_y <= 0.85:
                        condition_c_middle = True
                    elif id == 16 and 0.25 <= normalized_relative_x <= 0.4 and 0.7 <= normalized_relative_y <= 1:
                        condition_c_ring = True

                    # D코드
                    if id == 8 and 0.5 <= normalized_relative_x <= 0.6 and 0.4 <= normalized_relative_y <= 0.55:
                        condition_d_index = True
                    elif id == 12 and 0.4 <= normalized_relative_x <= 0.5 and 0.1 <= normalized_relative_y <= 0.25:
                        condition_d_middle = True
                    elif id == 16 and 0.25 <= normalized_relative_x <= 0.4 and 0.15 <= normalized_relative_y <= 0.3:
                        condition_d_ring = True

                    # E코드
                    if id == 8 and 0.7 <= normalized_relative_x <= 0.8 and 0.4 <= normalized_relative_y <= 0.55:
                        condition_e_index = True
                    elif id == 12 and 0.55 <= normalized_relative_x <= 0.65 and 0.75 <= normalized_relative_y <= 0.9:
                        condition_e_middle = True
                    elif id == 16 and 0.4 <= normalized_relative_x <= 0.5 and 0.6 <= normalized_relative_y <= 0.75:
                        condition_e_ring = True

                    # G코드
                    if id == 12 and 0.45 <= normalized_relative_x <= 0.55 and 0.8 <= normalized_relative_y <= 1.1:
                        condition_g_index = True
                    elif id == 16 and 0.25 <= normalized_relative_x <= 0.35 and 0.9 <= normalized_relative_y <= 1.1:
                        condition_g_middle = True
                    elif id == 20 and 0.2 <= normalized_relative_x <= 0.40 and 0 <= normalized_relative_y <= 0.3:
                        condition_g_ring = True

                    # A코드
                    if id == 8 and 0.6 <= normalized_relative_x <= 0.7 and 0.6 <= normalized_relative_y <= 0.8:
                        condition_a_index = True
                    elif id == 12 and 0.5 <= normalized_relative_x <= 0.6 and 0.5 <= normalized_relative_y <= 0.7:
                        condition_a_middle = True
                    elif id == 16 and 0.45 <= normalized_relative_x <= 0.55 and 0.2 <= normalized_relative_y <= 0.4:
                        condition_a_ring = True

    is_c_code_correct = condition_c_index and condition_c_middle and condition_c_ring
    is_d_code_correct = condition_d_index and condition_d_middle and condition_d_ring
    is_e_code_correct = condition_e_index and condition_e_middle and condition_e_ring
    is_g_code_correct = condition_g_index and condition_g_middle and condition_g_ring
    is_a_code_correct = condition_a_index and condition_a_middle and condition_a_ring

    # if condition_c_index and condition_c_middle and condition_c_ring:
    #     cv2.putText(frame, "C code correct", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # if condition_d_index and condition_d_middle and condition_d_ring:
    #     cv2.putText(frame, "D code correct", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # if condition_e_index and condition_e_middle and condition_e_ring:
    #     cv2.putText(frame, "E code correct", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # if condition_g_index and condition_g_middle and condition_g_ring:
    #     cv2.putText(frame, "G code correct", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # if condition_a_index and condition_a_middle and condition_a_ring:
    #     cv2.putText(frame, "A code correct", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    emit('response_back', {'image': jpg_as_text, 'hand_detected': hand_detected, 'hand_inside_box': hand_inside_box, 
                           'is_c_code_correct': is_c_code_correct,
                           'is_d_code_correct': is_d_code_correct,
                           'is_e_code_correct': is_e_code_correct,
                           'is_g_code_correct': is_g_code_correct,
                           'is_a_code_correct': is_a_code_correct,
                           'feedback': ""})

@socketio.on('request_feedback')
def handle_feedback_request(chord):
    global last_finger_positions
    feedback = generate_feedback(chord, last_finger_positions[chord])
    emit('feedback_response', {'feedback': feedback})

@socketio.on('audio_data')
def handle_audio(data):
    # Decode the base64 data
    audio_data = base64.b64decode(data)
    
    # Convert the byte data to a numpy array of the correct dtype
    audio_data = np.frombuffer(audio_data, dtype=np.float32)
    
    # Preprocess the audio data
    features = preprocess_audio(audio_data, scaler)
    
    # Predict the chord
    predicted_chord = predict_chord(model, features, encoder)
    
    # Emit the prediction result back to the client
    emit('audio_prediction', {'predicted_chord': predicted_chord})

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data = request.get_json()
        name = data['name']
        username = data['username']
        password = data['password']

        db = get_db()
        cursor = db.execute('SELECT id FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()

        if user is not None:
            return jsonify({'success': False, 'message': '이미 존재하는 Username입니다!'})

        hashed_password = generate_password_hash(password)
        db.execute('INSERT INTO users (name, username, password) VALUES (?, ?, ?)',
                   (name, username, hashed_password))
        db.commit()

        return jsonify({'success': True, 'message': '회원가입이 성공적으로 완료되었습니다!'})
    return render_template('signup.html', login_url=url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data['username']
        password = data['password']

        db = get_db()
        cursor = db.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()

        if user is None or not check_password_hash(user['password'], password):
            return jsonify({'success': False, 'message': 'Username 또는 비밀번호가 잘못 입력되었습니다!'})

        session['user_id'] = user['id']
        session['username'] = user['username']
        return jsonify({'success': True, 'message': '로그인 되었습니다!'})
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
