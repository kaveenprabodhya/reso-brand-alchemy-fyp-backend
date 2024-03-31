from flask import Flask, jsonify, request, Response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from datetime import datetime
from flask_socketio import SocketIO, emit
from emotion_detector import analyze_emotion_frame_async, process_frame_for_motion
from generate_image import generate_image_from_text, download_images
from PIL import Image as PILImage
from threading import Lock
from flask import session
from flask_jwt_extended import JWTManager, create_access_token, verify_jwt_in_request, get_jwt_identity, jwt_required
from datetime import timedelta, datetime
from collections import defaultdict
from bson.objectid import ObjectId
from pymongo import MongoClient, DESCENDING
from bson.binary import Binary
import io
import base64
import cv2
import numpy as np
import re
import uuid
import os

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'your_jwt_secret_key'
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/resobrandalchemy'
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE'] = True
CORS(app, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins="*", manage_session=False)
jwt = JWTManager(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)

# MongoDB setup
mongo = MongoClient(app.config['MONGO_URI'])
db = mongo.resobrandalchemy
users_collection = db.users
images_collection = db.images

emotion_analysis_results = []


def validate_email(email):
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if not re.fullmatch(regex, email):
        return False
    return True


def validate_password(password):
    if len(password) < 8 or not re.search("[a-zA-Z]", password) or not re.search("[0-9]", password):
        return False
    return True


class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.email = user_data['email']
        self.password = user_data['password']
        self.first_name = user_data.get('first_name', '')
        self.last_name = user_data.get('last_name', '')

    @staticmethod
    def get(user_id):
        user_data = db.users.find_one({"_id": ObjectId(user_id)})
        if not user_data:
            return None
        return User(user_data)


@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)


@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.json
    if not all(field in data for field in ['email', 'password', 'firstName', 'lastName']):
        return jsonify({'message': 'Missing data'}), 400
    if not validate_email(data['email']):
        return jsonify({'message': 'Invalid email format'}), 400
    if not validate_password(data['password']):
        return jsonify({'message': 'Password requirements not met'}), 400

    if db.users.find_one({'email': data['email']}):
        return jsonify({'message': 'Email already exists'}), 409

    hashed_password = bcrypt.generate_password_hash(
        data['password']).decode('utf-8')
    user_id = db.users.insert_one({
        'email': data['email'],
        'password': hashed_password,
        'first_name': data['firstName'],
        'last_name': data['lastName']
    }).inserted_id

    new_user = User.get(user_id)
    login_user(new_user, remember=True)

    additional_claims = {'email': new_user.email,
                         'first_name': new_user.first_name, 'last_name': new_user.last_name}

    access_token = create_access_token(identity=str(
        new_user.id), expires_delta=timedelta(hours=1), additional_claims=additional_claims)

    return jsonify({'message': 'User registered successfully', 'access_token': access_token}), 201


@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.json
    user = db.users.find_one({"email": data.get("email")})
    if user and bcrypt.check_password_hash(user["password"], data.get("password")):
        user_obj = User(user)
        login_user(user_obj, remember=True)
        payload = {'email': user_obj.email,
                   'first_name': user_obj.first_name, 'last_name': user_obj.last_name}
        access_token = create_access_token(identity=str(
            user_obj.id), expires_delta=timedelta(days=1), additional_claims=payload)
        return jsonify({"message": "Logged in successfully", "access_token": access_token}), 200
    return jsonify({"message": "Invalid email or password"}), 401


@app.route('/api/auth/logout', methods=['POST'])
@jwt_required()
def logout():
    logout_user()
    return jsonify({"message": "Logged out successfully"}), 200


@app.route('/api/auth/delete-account', methods=['POST'])
@jwt_required()
def delete_account():
    user_id = get_jwt_identity()
    db.users.delete_one({"_id": ObjectId(user_id)})
    logout_user()
    return jsonify({"message": "Account deleted successfully"}), 200


def add_images_to_mongodb(img_paths, metadata, batch_id):
    img_ids = []
    for path in img_paths:
        with open(path, 'rb') as file:
            content = Binary(file.read())
            metadata_copy = metadata.copy()
            metadata_copy.update({
                'image': content,
                'batch_id': batch_id  # Include the batch ID in each document
            })
            img_id = images_collection.insert_one(metadata_copy).inserted_id
            img_ids.append(img_id)
    return img_ids


@app.route('/api/image/generate-image', methods=['POST'])
@jwt_required()
def generate_image():
    current_user_id = get_jwt_identity()
    data = request.json
    brand_name = data.get("brandName")
    colors_captured = data.get("colors")

    print(brand_name)
    print(colors_captured)

    if not brand_name:
        return jsonify({"error": "Brand name is required."}), 400

    if not colors_captured:
        return jsonify({"error": "Captured colors are required."}), 400

    image_urls = generate_image_from_text(brand_name, colors_captured)
    batch_id = str(uuid.uuid4())
    img_paths = download_images(image_urls, batch_id)

    metadata = {
        "user_id": ObjectId(current_user_id),
        "brand_name": brand_name,
        "creation_date": datetime.utcnow()
    }
    img_ids = add_images_to_mongodb(img_paths, metadata, batch_id)
    base_url = request.host_url.rstrip('/')
    local_urls = [f'{base_url}/image/{str(img_id)}' for img_id in img_ids]

    return jsonify({
        'message': 'Images generated and saved successfully',
        'local_urls': local_urls,
    }), 201


@app.route('/image/<image_id>')
def get_image(image_id):
    image_data = images_collection.find_one({"_id": ObjectId(image_id)})
    if not image_data or 'image' not in image_data:
        return jsonify({"error": "Image not found"}), 404
    return Response(image_data['image'], mimetype='image/jpeg')


@app.route('/api/image/get-history', methods=['GET'])
@jwt_required()
def get_image_history_for_user():
    current_user_id = get_jwt_identity()
    user_images_query = {"user_id": ObjectId(current_user_id)}

    # Fetch images and sort by creation_date, brand_name, and batch_id in a meaningful order
    user_images = images_collection.find(user_images_query).sort(
        [("creation_date", DESCENDING), ("brand_name", 1), ("batch_id", 1)])

    # Initialize a structure to hold organized image data
    organized_images = []

    base_url = request.host_url.rstrip('/')
    for img in user_images:
        # Construct the image URL
        image_url = f'{base_url}/image/{str(img["_id"])}'

        # Use batch_id as an identifier, if available, or a placeholder if not
        batch_id = img.get("batch_id", "Unknown")

        # Find an existing entry for the brand_name, created, and batch_id or create a new one
        existing_entry = next((item for item in organized_images if item["brandName"] == img["brand_name"] and
                               item["created"] == img["creation_date"].strftime("%d/%m/%Y") and
                               item["id"] == batch_id), None)

        if existing_entry:
            # If found, append the new image URL to the imgs list of the existing entry
            existing_entry["imgs"].append(image_url)
        else:
            # If not found, create a new entry for this combination of brand_name, creation_date, and batch_id
            organized_images.append({
                "id": batch_id,
                "brandName": img["brand_name"],
                "imgs": [image_url],
                "created": img["creation_date"].strftime("%d/%m/%Y")
            })

    if not organized_images:
        return jsonify({"message": "No images found for the user"}), 404

    return jsonify(organized_images), 200


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
        'happy': '#FFEA00',
        'sad': '#0077B6',
        'angry': '#D32F2F',
        'disgusted': '#388E3C',
        'fearful': '#9C27B0',
        'neutral': '#9E9E9E',
        'surprised': '#FF9800',
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
    socketio.run(app, debug=True)
