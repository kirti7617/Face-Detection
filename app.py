from flask import Flask, render_template, Response
import cv2
from deepface import DeepFace

app = Flask(__name__)


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


emoji_images = {
    "angry": cv2.imread("emojis/angry.png", cv2.IMREAD_UNCHANGED),
    "disgust": cv2.imread("emojis/disgust.png", cv2.IMREAD_UNCHANGED),
    "fear": cv2.imread("emojis/fear.png", cv2.IMREAD_UNCHANGED),
    "happy": cv2.imread("emojis/happy.png", cv2.IMREAD_UNCHANGED),
    "sad": cv2.imread("emojis/sad.png", cv2.IMREAD_UNCHANGED),
    "surprise": cv2.imread("emojis/surprise.png", cv2.IMREAD_UNCHANGED),
    "neutral": cv2.imread("emojis/neutral.png", cv2.IMREAD_UNCHANGED)
}

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    
    """Overlay `img_overlay` onto `img` at (x, y) with alpha channel from `alpha_mask`."""
    h, w = img_overlay.shape[:2]

    
    y1, y2 = max(0, y), min(img.shape[0], y + h)
    x1, x2 = max(0, x), min(img.shape[1], x + w)

    y1o, y2o = max(0, -y), min(h, img.shape[0] - y)
    x1o, x2o = max(0, -x), min(w, img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, None] / 255.0

    img_crop[:] = (1.0 - alpha) * img_crop + alpha * img_overlay_crop


def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
        faces = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = rgb_frame[y:y+h, x:x+w]
            result = DeepFace.analyze(
                face_roi, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            emoji_img = emoji_images.get(emotion)

            if emoji_img is not None:
                emoji_resized = cv2.resize(
                    emoji_img, (w, h), interpolation=cv2.INTER_AREA)
                overlay_image_alpha(
                    frame, emoji_resized[:, :, :3], x, y, emoji_resized[:, :, 3])

        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)