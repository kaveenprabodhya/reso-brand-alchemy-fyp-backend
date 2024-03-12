from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from datetime import timedelta
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from emotion_detector import analyze_emotion_frame
from generate_image import generate_image_from_text
import io
from PIL import Image as PILImage
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app)  # Enable CORS for all domains
socketio = SocketIO(app, cors_allowed_origins="*")

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)

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

@app.route('/api/image/video-emotion', methods=['POST'])
@login_required
def video_emotion():
    # Placeholder for video stream processing and emotion detection
    emotions, colors = analyze_emotion_frame(request.data)
    return jsonify({'emotions': emotions, 'colors': colors}), 200

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
    emit('my response', {'data': 'Connected'})

def convert_image(data):
    """
    Convert the received image data from binary to a PIL Image.
    This function assumes the binary data is a JPEG image.
    """
    image_stream = io.BytesIO(data)
    image_stream.seek(0)
    image = PILImage.open(image_stream)
    return image

@socketio.on('stream_frame')
def handle_video_frame(data):
    # Process the frame for emotion detection
    # `data` should contain the necessary information to process the frame
    image = convert_image(data)
    emotions = analyze_emotion_frame(image)
    # Emit detected emotions back to the client
    emit('emotion_data', {'emotions': {"emotion": emotions, "colors": "red"}})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    socketio.run(app, debug=True)
