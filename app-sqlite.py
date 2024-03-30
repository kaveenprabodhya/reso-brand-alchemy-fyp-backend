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
from flask_jwt_extended import JWTManager, create_access_token, verify_jwt_in_request, get_jwt_identity, jwt_required
from datetime import timedelta, datetime
from collections import defaultdict
import io
import base64
import cv2
import numpy as np
import re
# import os

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'your_jwt_secret_key'
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///resobrandalchemy.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE'] = True
CORS(app, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins="*", manage_session=False)
jwt = JWTManager(app)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)

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
    image = db.Column(db.String(300), nullable=False)
    creation_date = db.Column(
        db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f'<Image {self.id}, {self.user_id}, {self.image}, {self.creation_date}>'


def validate_email(email):
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if not re.fullmatch(regex, email):
        return False
    return True


def validate_password(password):
    if len(password) < 8 or not re.search("[a-zA-Z]", password) or not re.search("[0-9]", password):
        return False
    return True


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.json
    required_fields = ['email', 'password', 'firstName', 'lastName']
    if not all(field in data for field in required_fields):
        return jsonify({'message': 'Missing data'}), 400

    if not validate_email(data['email']):
        return jsonify({'message': 'Invalid email format'}), 400
    if not validate_password(data['password']):
        return jsonify({'message': 'Password must be at least 8 characters long and include both letters and numbers'}), 400

    user = User.query.filter_by(email=data['email']).first()
    if user:
        return jsonify({'message': 'User already exists'}), 409

    hashed_password = bcrypt.generate_password_hash(
        data['password']).decode('utf-8')
    new_user = User(first_name=data['firstName'], last_name=data['lastName'],
                    email=data['email'], password=hashed_password)
    db.session.add(new_user)
    db.session.commit()

    login_user(new_user, remember=True)

    additional_claims = {'email': new_user.email,
                         'first_name': new_user.first_name, 'last_name': new_user.last_name}
    access_token = create_access_token(identity=str(
        new_user.id), expires_delta=timedelta(hours=1), additional_claims=additional_claims)

    session['user_id'] = new_user.id

    return jsonify({'message': 'User created, logged in successfully, and token generated', 'access_token': access_token}), 201


@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.json
    if not data or 'email' not in data or 'password' not in data:
        return jsonify({'message': 'Missing data'}), 400

    user = User.query.filter_by(email=data['email']).first()
    if user and bcrypt.check_password_hash(user.password, data['password']):
        login_user(user, remember=True)
        payload = {'email': user.email,
                   'first_name': user.first_name, 'last_name': user.last_name}
        access_token = create_access_token(identity=str(
            user.id), expires_delta=timedelta(hours=1), additional_claims=payload)
        session['user_id'] = user.id
        return jsonify({'message': 'Logged in successfully', 'access_token': access_token}), 200
    return jsonify({'message': 'Invalid email or password'}), 401


@app.route('/api/auth/logout', methods=['POST'])
@jwt_required()
def logout():
    return jsonify({'message': 'Logged out successfully'}), 200


@app.route('/api/auth/delete-account', methods=['POST'])
@jwt_required()
def delete_account():
    user_id = get_jwt_identity()  # Get the user ID from the JWT
    user = User.query.get(user_id)
    if user:
        db.session.delete(user)
        db.session.commit()

        logout_user()

        return jsonify({'message': 'User deleted successfully'}), 200
    else:
        return jsonify({'message': 'User not found'}), 404


@app.route('/api/image/generate-image', methods=['POST'])
@jwt_required()
def generate_image():
    data = request.json
    images = generate_image_from_text(data['text'])

    created_images_info = []

    for image in images:
        new_image = Image(user_id=current_user.id, image=image)
        db.session.add(new_image)
        db.session.commit()

        # Add the image path and creation date to our list
        created_images_info.append({'image': image})

    # Return the list of created images
    return jsonify(created_images_info), 201


@app.route('/api/image/get-history')
@jwt_required()
def get_image_history_for_user():
    user_id = get_jwt_identity()  # Use JWT identity to fetch the current user's ID
    user_images = Image.query.filter_by(user_id=user_id).order_by(
        Image.creation_date.desc()).all()

    if user_images:
        images_by_date = defaultdict(list)
        for img in user_images:
            date_key = img.creation_date.strftime("%d/%m/%Y")
            images_by_date[date_key].append(img.image)

        data = []
        for creation_date, images in images_by_date.items():
            data.append({
                'date': creation_date,  # The date when the images were added
                'imgs': images,  # List of image URLs added on this date
            })

        # data = [
        #     {
        #         'id': 1,
        #         'brandName': "RBV logo",
        #         'imgs': [
        #             "https://mbluxury1.s3.amazonaws.com/2022/02/25172616/chanel-1.jpg",
        #             "https://cdn.mos.cms.futurecdn.net/JYMtGzLr6f2RGuHkaxJjwD-1200-80.jpg",
        #             "https://lh3.googleusercontent.com/3bXLbllNTRoiTCBdkybd1YzqVWWDRrRwJNkVRZ3mcf7rlydWfR13qJlCSxJRO8kPe304nw1jQ_B0niDo56gPgoGx6x_ZOjtVOK6UGIr3kshpmTq46pvFObfJ2K0wzoqk36MWWSnh0y9PzgE7PVSRz6Y",
        #             "https://1000logos.net/wp-content/uploads/2016/10/Batman-Logo-1939.png",
        #             "https://t3.ftcdn.net/jpg/04/27/44/04/360_F_427440465_7Fgl0hGr3JTjKOVSoUgsRlcL6kUEoTdl.jpg",
        #         ],
        #         'created': "12/12/2022",
        #     },
        #     {
        #         'id': 2,
        #         'brandName': "Can logo",
        #         'imgs': [
        #             "https://img.freepik.com/free-vector/bird-colorful-logo-gradient-vector_343694-1365.jpg?size=338&ext=jpg&ga=GA1.1.1803636316.1711814400&semt=sph",
        #             "https://img.freepik.com/free-vector/gradient-quill-pen-design-template_23-2149837194.jpg",
        #             "https://marketplace.canva.com/EAFauoQSZtY/1/0/1600w/canva-brown-mascot-lion-free-logo-qJptouniZ0A.jpg",
        #             "https://cdn.pixabay.com/photo/2017/03/16/21/18/logo-2150297_640.png",
        #             "https://images.fastcompany.net/image/upload/w_596,c_limit,q_auto:best,f_auto/fc/3034007-inline-i-applelogo.jpg",
        #         ],
        #         'created': "23/02/2024",
        #     },
        # ]

        return jsonify(data), 200
    else:
        return jsonify({'message': 'No images found for the user'}), 404


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


def verify_token(auth_token):
    try:
        with app.test_request_context(headers={"Authorization": f"Bearer {auth_token}"}):
            verify_jwt_in_request()
            return True
    except Exception as e:
        print(f"Token verification error: {e}")
        return False


def decode_auth_token(auth_token):
    try:
        with app.test_request_context(headers={"Authorization": f"Bearer {auth_token}"}):
            user_id = get_jwt_identity()
            return user_id
    except Exception as e:
        print(f"Token decoding error: {e}")
        return None


def get_user_id_from_session():
    """
    Retrieves the current user's ID from the session using Flask-Login's current_user.
    Returns:
        The user's ID if a user is logged in, None otherwise.
    """
    if current_user.is_authenticated:
        return current_user.get_id()
    return None


@socketio.on('connect')
def socket_server_connect():
    emit('server_connect_response', {'status': 'Connected'})


def convert_image(data):
    if isinstance(data, str):
        image_data = base64.b64decode(data.split(",")[-1])
    else:
        image_data = data
    image_stream = io.BytesIO(image_data)
    image_stream.seek(0)
    image = PILImage.open(image_stream)
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)


@socketio.on('stream_frame')
def handle_video_frame(data):
    global analysis_in_progress, reference_frame
    frame = convert_image(data)
    # save_directory = "C:\\Users\\kaveenThivanka\\Desktop\\fyp\\images"
    # if not os.path.exists(save_directory):
    #     os.makedirs(save_directory)
    if reference_frame is None:
        update_reference_frame(frame)
        emit('frame_status', {'status': 'reference_frame_set'})
        return
    if process_frame_for_motion(frame, reference_frame):
        update_reference_frame(frame)
        with analysis_lock:
            analysis_in_progress += 1
        analyze_emotion_frame_async(frame, callback=emotion_analysis_callback)
        # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # cv2.imwrite(os.path.join(save_directory,
        #             f"frame_{timestamp}.png"), frame)


def emotion_analysis_callback(emotion):
    global analysis_in_progress, streaming_stopped, emotion_analysis_results
    with analysis_lock:
        emotion_analysis_results.append(emotion)
        analysis_in_progress -= 1
        if analysis_in_progress == 0:
            send_emotion_analysis_results()


def send_emotion_analysis_results():
    global emotion_analysis_results
    sanitized_results = process_emotion_results(emotion_analysis_results)
    print(sanitized_results)
    socketio.emit('emotion_analysis_results', {'results': sanitized_results})
    emotion_analysis_results.clear()


def process_emotion_results(results):
    sanitized_results = {}
    previous_emotion = None
    for result in results:
        current_emotion = result
        if current_emotion != previous_emotion:
            color = get_emotion_color(current_emotion)
            sanitized_results[current_emotion] = color
            previous_emotion = current_emotion

    return sanitized_results


def get_emotion_color(emotion):
    emotion_color_mapping = {
        'happy': 'yellow',
        'sad': 'blue',
        'angry': 'red',
        'disgusted': 'green',
        'fearful': 'purple',
        'neutral': 'gray',
        'surprised': 'orange',
    }
    return emotion_color_mapping.get(emotion, 'black')


@socketio.on('stop_stream')
def handle_stop_stream():
    global streaming_stopped
    streaming_stopped = True


@socketio.on('disconnect')
def handle_disconnect():
    pass


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    socketio.run(app, debug=True)
