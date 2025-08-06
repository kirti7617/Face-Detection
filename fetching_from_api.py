import cv2
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from deepface import DeepFace
from flask import Flask, render_template, Response
import cairosvg
# Initialize Flask app
app = Flask(__name__)

# Function to convert emotion into OpenMoji Unicode
def get_emoji_unicode(emotion):
    """Convert detected emotion to OpenMoji Unicode for fetching the correct emoji."""
    emoji_map = {
        "angry": "1F620",      # 😠
        "disgust": "1F922",    # 🤢
        "fear": "1F628",       # 😨
        "happy": "1F600",      # 😀
        "sad": "1F622",        # 😢
        "surprise": "1F632",   # 😲
        "neutral": "1F610"     # 😐
    }
    return emoji_map.get(emotion, "1F610")  # Default to neutral emoji

# Function to fetch emoji from OpenMoji API
def fetch_emoji(emoji_unicode):
    """Fetch emoji from OpenMoji, convert SVG to PNG, and return as numpy array."""
    base_url = "https://openmoji.org/data/color/svg/"
    emoji_url = f"{base_url}{emoji_unicode.upper()}.svg"

    try:
        response = requests.get(emoji_url)
        if response.status_code == 200:
            # Convert SVG to PNG
            png_data = BytesIO()
            cairosvg.svg2png(bytestring=response.content, write_to=png_data)
            png_data.seek(0)

            # Open as RGBA image
            emoji_img = Image.open(png_data).convert("RGBA")
            emoji_img = emoji_img.resize((72, 72))  # Resize to fit faces
            emoji_np = np.array(emoji_img)

            return emoji_np  # Return as numpy array for OpenCV processing
        else:
            print(f"Failed to fetch emoji: {emoji_url} (Status Code: {response.status_code})")
    except Exception as e:
        print(f"Error fetching emoji: {e}")

    return None  # Return None if fetching fails

# Function to overlay emoji on the detected face
def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Overlay `img_overlay` onto `img` at (x, y) using `alpha_mask` for transparency."""
    h, w = img_overlay.shape[:2]

    # Ensure emoji fits within frame
    y1, y2 = max(0, y), min(img.shape[0], y + h)
    x1, x2 = max(0, x), min(img.shape[1], x + w)

    y1o, y2o = max(0, -y), min(h, img.shape[0] - y)
    x1o, x2o = max(0, -x), min(w, img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]

    # Fix alpha mask dimensions
    alpha_mask = cv2.resize(alpha_mask, (w, h), interpolation=cv2.INTER_AREA)
    alpha_mask = alpha_mask[:, :, np.newaxis]  # Shape (h, w, 1)
    alpha_mask = np.repeat(alpha_mask, 3, axis=2)  # Shape (h, w, 3)

    # Ensure dimensions match before blending
    if img_crop.shape != alpha_mask.shape or img_crop.shape != img_overlay_crop.shape:
        print(f"Shape Mismatch! Image: {img_crop.shape}, Overlay: {img_overlay_crop.shape}, Alpha: {alpha_mask.shape}")
        return

    # Blend the emoji with the frame
    img_crop[:] = (1.0 - alpha_mask) * img_crop + alpha_mask * img_overlay_crop

# Function to generate frames for video stream
def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        detected_faces = faces.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in detected_faces:
            face_roi = frame[y:y+h, x:x+w]
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']

            # Fetch emoji dynamically from OpenMoji
            emoji_unicode = get_emoji_unicode(emotion)
            emoji_img = fetch_emoji(emoji_unicode)

            if emoji_img is not None:
                # Resize emoji to match face size
                emoji_resized = cv2.resize(emoji_img[:, :, :3], (w, h), interpolation=cv2.INTER_AREA)
                alpha_mask = cv2.resize(emoji_img[:, :, 3], (w, h), interpolation=cv2.INTER_AREA)

                # Fix alpha mask dimensions
                alpha_mask = alpha_mask[:, :, np.newaxis]  # Shape (h, w, 1)
                alpha_mask = np.repeat(alpha_mask, 3, axis=2)  # Shape (h, w, 3)

                # Overlay emoji onto the frame
                overlay_image_alpha(frame, emoji_resized, x, y, alpha_mask)

        # Encode and yield frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
