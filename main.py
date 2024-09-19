from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# Initialize the camera
camera = cv2.VideoCapture(0)

# Load all the cascade files for ISL letters (A to Z)
cascade_files = {
    'A': 'Haar Cascades/A.xml',
    'B': 'Haar Cascades/B.xml',
    'C': 'Haar Cascades/C.xml',
    'D': 'Haar Cascades/D.xml',
    'E': 'Haar Cascades/E.xml',
    'F': 'Haar Cascades/F.xml',
    'G': 'Haar Cascades/G.xml',
    'H': 'Haar Cascades/H.xml',
    'I': 'Haar Cascades/I.xml',
    'J': 'Haar Cascades/J.xml',
    'K': 'Haar Cascades/K.xml',
    'L': 'Haar Cascades/L.xml',
    # Add more cascade files for each letter or number
}

# Create a dictionary to hold all the loaded cascade classifiers
cascades = {letter: cv2.CascadeClassifier(cascade_path) for letter, cascade_path in cascade_files.items()}

def generate_frames():
    while True:
        # Read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convert the frame to grayscale (required by Haar cascades)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Define the ROI (Region of Interest) where you want detection to occur
            height, width = gray.shape
            roi_x_start = int(width * 0.3)  # X-coordinate start (30% from the left)
            roi_y_start = int(height * 0.3)  # Y-coordinate start (30% from the top)
            roi_x_end = int(width * 0.7)     # X-coordinate end (70% from the left)
            roi_y_end = int(height * 0.7)    # Y-coordinate end (70% from the top)

            # Crop the region of interest (ROI) from the grayscale frame
            roi_gray = gray[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
            roi_color = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

            # Loop through all the cascades and detect corresponding actions within the ROI
            for letter, cascade in cascades.items():
                detections = cascade.detectMultiScale(roi_gray, 1.1, 5)

                # Draw rectangles around detected actions and label them
                for (x, y, w, h) in detections:
                    # Adjust the coordinates to the full frame
                    cv2.rectangle(roi_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(roi_color, letter, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Encode the frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
