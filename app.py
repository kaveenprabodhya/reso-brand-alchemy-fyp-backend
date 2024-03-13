from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from datetime import datetime
from flask_socketio import SocketIO, emit
from emotion_detector import analyze_emotion_frame_async, process_frame_for_motion
from generate_image import generate_image_from_text
from PIL import Image as PILImage
from threading import Lock
from flask import session
import io
import base64
import cv2
import numpy as np
import re

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///resobrandalchemy.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins="*", manage_session=False)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)

user_sids = {}

emotion_analysis_results = []

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(30), unique=True, nullable=False)
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

def validate_email(email):
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if not re.fullmatch(regex, email):
        return False
    return True

def validate_password(password):
    # Example: Minimum 8 characters, at least one letter and one number
    if len(password) < 8 or not re.search("[a-zA-Z]", password) or not re.search("[0-9]", password):
        return False
    return True

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.json
    # Validate presence of required fields
    required_fields = ['email', 'password', 'first_name', 'last_name']
    if not all(field in data for field in required_fields):
        return jsonify({'message': 'Missing data'}), 400

    # Validate email and password
    if not validate_email(data['email']):
        return jsonify({'message': 'Invalid email format'}), 400
    if not validate_password(data['password']):
        return jsonify({'message': 'Password must be at least 8 characters long and include both letters and numbers'}), 400

    # Check for existing user5
    user = User.query.filter_by(email=data['email']).first()
    if user:
        return jsonify({'message': 'User already exists'}), 409

    # Proceed with user creation
    hashed_password = bcrypt.generate_password_hash(data['password']).decode('utf-8')
    new_user = User(first_name=data['first_name'], last_name=data['last_name'], email=data['email'], password=hashed_password)
    db.session.add(new_user)
    db.session.commit()

    # Log the user in automatically after registration
    login_user(new_user, remember=True)
    session['user_id'] = new_user.id  # Again, explicitly setting user ID in the session, if required

    return jsonify({'message': 'User created and logged in successfully', 'user_id': new_user.id}), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.json
    if not data or 'email' not in data or 'password' not in data:
        return jsonify({'message': 'Missing data'}), 400

    user = User.query.filter_by(email=data['email']).first()
    if user and bcrypt.check_password_hash(user.password, data['password']):
        login_user(user, remember=True)
        # Flask-Login handles session management, but you can add more session data as needed
        session['user_id'] = user.id  # Explicitly setting user ID in the session, if required
        return jsonify({'message': 'Logged in successfully', 'user_id': user.id}), 200
    return jsonify({'message': 'Invalid email or password'}), 401

@app.route('/api/auth/current-user', methods=['GET'])
@login_required
def get_current_user():
    # Assuming current_user has an attribute 'to_dict' method that returns its properties as a dictionary
    # If not, you'll need to manually construct this dictionary based on your User model
    user_data = current_user.to_dict() if hasattr(current_user, 'to_dict') else {
        'id': current_user.id,
        'email': current_user.email,
        'first_name': current_user.first_name,
        'last_name': current_user.last_name
    }
    return jsonify(user_data), 200

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
def socket_server_connect():
    if current_user.is_authenticated:
        user_sids[current_user.get_id()] = request.sid
        emit('server_connect_response', {'status': 'Connected', 'user_id': current_user.get_id()})
    else:
        emit('server_connect_response', {'status': 'Unauthorized'})

def convert_image(data):
    if isinstance(data, str):
        # If the data is indeed a Base64 string, ensure it's correctly formatted
        image_data = base64.b64decode(data.split(",")[-1])  # Consider splitting if necessary
    else:
        # Handle raw binary data directly
        image_data = data
    image_stream = io.BytesIO(image_data)
    image_stream.seek(0)
    image = PILImage.open(image_stream)
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

reference_frame = None
analysis_in_progress = 0
analysis_lock = Lock()
streaming_stopped = False

def get_reference_frame():
    global reference_frame
    return reference_frame

def update_reference_frame(new_frame):
    global reference_frame
    reference_frame = new_frame

@socketio.on('stream_frame')
def handle_video_frame(data):
    global analysis_in_progress, reference_frame
    frame = convert_image(data)  # Frame is now a grayscale image

    user_id = current_user.get_id() if current_user.is_authenticated else None
    print(user_id)

    if not user_id:
        # Handle unauthenticated user scenario, perhaps by skipping processing or notifying the client
        emit('error', {'message': 'User not authenticated'})
        return

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
        with analysis_lock:
            analysis_in_progress += 1
        # Assume analyze_emotion_frame is an asynchronous operation and accepts a callback
        analyze_emotion_frame_async(frame, user_id=user_id, callback=emotion_analysis_callback)

def emotion_analysis_callback(emotion, user_id):
    global analysis_in_progress, streaming_stopped
    with analysis_lock:
        emotion_analysis_results.append((emotion, user_id))
        print(emotion)
        analysis_in_progress -= 1
        if streaming_stopped and analysis_in_progress == 0:
            send_emotion_analysis_results(user_id)

def send_emotion_analysis_results():
    global emotion_analysis_results
    # Process results and emit them to the correct user based on user_id
    for emotion, user_id in emotion_analysis_results:
        sanitized_results = process_emotion_results([emotion])  # Assuming process_emotion_results is adapted for single results
        sid = user_sids.get(str(user_id))
        if sid:
            socketio.emit('emotion_analysis_results', {'results': sanitized_results}, room=sid)
        else:
            print(f"No active session for user ID {user_id}")
    emotion_analysis_results.clear()

@socketio.on('stop_stream')
def handle_stop_stream():
    global streaming_stopped
    streaming_stopped = True
    with analysis_lock:
        if analysis_in_progress == 0:
            send_emotion_analysis_results()


def process_emotion_results(results):
    sanitized_results = {}
    print(results)
    previous_emotion = None
    for result in results:
        current_emotion = result
        if current_emotion != previous_emotion:
            # Map the emotion to a color
            color = get_emotion_color(current_emotion)
            # Store the color in the sanitized_results dictionary
            sanitized_results[current_emotion] = color
            previous_emotion = current_emotion

    print(sanitized_results)

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

@socketio.on('disconnect')
def handle_disconnect():
    user_id_to_remove = None
    for user_id, sid in user_sids.items():
        if sid == request.sid:
            user_id_to_remove = user_id
            break
    if user_id_to_remove:
        del user_sids[user_id_to_remove]

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    socketio.run(app, debug=True)
