from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from datetime import timedelta
from datetime import datetime
from flask_socketio import SocketIO, emit
from emotion_detector import analyze_emotion_frame, process_frame_for_motion
from generate_image import generate_image_from_text
import io
from PIL import Image as PILImage
import base64
import cv2
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)


emotion_analysis_results = []

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(30), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    images = db.relationship('Image', backref='user', lazy=True)

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(300), nullable=False)
    creation_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f'<Image {self.id}, {self.user_id}, {self.image_path}, {self.creation_date}>'    

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.json
    if not data or 'username' not in data or 'password' not in data or 'first_name' not in data or 'last_name' not in data:
        return jsonify({'message': 'Missing data'}), 400

    user = User.query.filter_by(username=data['username']).first()
    if user:
        return jsonify({'message': 'User already exists'}), 409

    hashed_password = bcrypt.generate_password_hash(data['password']).decode('utf-8')
    new_user = User(first_name=data['first_name'], last_name=data['last_name'], username=data['username'], password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'message': 'User created successfully'}), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.json
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({'message': 'Missing data'}), 400

    user = User.query.filter_by(username=data['username']).first()
    if user and bcrypt.check_password_hash(user.password, data['password']):
        login_user(user, remember=True)
        return jsonify({'message': 'Logged in successfully'}), 200
    return jsonify({'message': 'Invalid username or password'}), 401

@app.route('/api/auth/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({'message': 'Logged out successfully'}), 200

@app.route('/api/auth/delete-account', methods=['POST'])
@login_required
def delete_account():
    user = User.query.get(current_user.id)
    if user:
        db.session.delete(user)
        db.session.commit()
        logout_user()
        return jsonify({'message': 'User deleted successfully'}), 200
    return jsonify({'message': 'User not found'}), 404

@app.route('/api/image/generate-image', methods=['POST'])
@login_required
def generate_image():
    data = request.json
    # Assume generate_image_from_text returns a path to the generated image
    image_path = generate_image_from_text(data['text'])
    
    new_image = Image(user_id=current_user.id, image_path=image_path)
    db.session.add(new_image)
    db.session.commit()
    return jsonify({'image_path': image_path, 'creation_date': new_image.creation_date.isoformat()}), 201

@socketio.on('connect')
def test_connect():
    emit('server_connect_response', {'status': 'Connected'})

def convert_image(data):
    image_data = base64.b64decode(data)
    image_stream = io.BytesIO(image_data)
    image_stream.seek(0)
    image = PILImage.open(image_stream)
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    return gray_image


reference_frame = None

def get_reference_frame():
    global reference_frame
    return reference_frame

def update_reference_frame(new_frame):
    global reference_frame
    reference_frame = new_frame

@socketio.on('stream_frame')
def handle_video_frame(data):
    global reference_frame
    frame = convert_image(data)  # Frame is now a grayscale image

    # If we don't have a reference frame yet, set the current frame as reference
    if reference_frame is None:
        update_reference_frame(frame)
        emit('frame_status', {'status': 'reference_frame_set'})
        return

    # Now use the motion detection function
    if process_frame_for_motion(frame, reference_frame):
        # If significant motion is detected, update the reference frame and proceed
        update_reference_frame(frame)
        # Placeholder for emotion detection logic
        # emotions = analyze_emotion_frame(frame)
        emotion = analyze_emotion_frame(frame)
        # Append detected emotions to the results list
        emotion_analysis_results.append(emotion)
    else:
        emit('frame_status', {'status': 'no_significant_motion'})

@socketio.on('stop_stream')
def handle_stop_stream():
    global emotion_analysis_results
    sanitized_results = {}
    previous_emotion = None
    for result in emotion_analysis_results:
        current_emotion = result["emotion"]
        if current_emotion != previous_emotion:
            # Map the emotion to a color
            color = get_emotion_color(current_emotion)
            # Store the color in the sanitized_results dictionary
            sanitized_results[current_emotion] = color
            previous_emotion = current_emotion

    print(sanitized_results)

    # Emit the sanitized emotion analysis results with their corresponding colors
    emit('emotion_analysis_results', {'results': sanitized_results})

    # Clear the results after sending
    emotion_analysis_results = []


def get_emotion_color(emotion):
    # Define a mapping of emotions to colors
    emotion_color_mapping = {
        'happy': 'yellow',
        'sad': 'blue',
        'angry': 'red',
        'disgusted': 'green',
        'fearful': 'purple',
        'neutral': 'gray',
        'surprised': 'orange',
    }
    # Return the color associated with the emotion, or 'black' if the emotion is unknown
    return emotion_color_mapping.get(emotion, 'black')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    socketio.run(app, debug=True)
