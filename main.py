import cv2
import datetime
import time
from deepface import DeepFace

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Timestamp for FPS
prev_time = time.time()

print("Starting webcam. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize (640x480) for performance
    resized_frame = cv2.resize(frame, (640, 480))

    try:
        # Analyze frame with DeepFace
        result = DeepFace.analyze(resized_frame, actions=['emotion'], enforce_detection=False)

        # Get bounding box for face
        region = result[0]['region']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Get emotion data
        emotions = result[0]['emotion']
        dominant_emotion = result[0]['dominant_emotion']
        confidence = emotions[dominant_emotion]

        # Display dominant emotion
        cv2.putText(resized_frame, f'Emotion: {dominant_emotion}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Emotion bars
        y0 = 60
        for emotion, score in emotions.items():
            bar_width = int(score * 1.5)
            cv2.rectangle(resized_frame, (20, y0), (20 + bar_width, y0 + 10), (0, 255, 0), -1)
            cv2.putText(resized_frame, f"{emotion}: {int(score)}%", (25 + bar_width, y0 + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y0 += 15

    except Exception as e:
        print("Error:", e)

    # Timestamp
    now = datetime.datetime.now().strftime("%H:%M:%S")
    cv2.putText(resized_frame, f"Time: {now}", (480, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # FPS Counter
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(resized_frame, f"FPS: {int(fps)}", (480, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Show the frame
    cv2.imshow("Emotion Recognition", resized_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
