import smtplib
import dlib
import numpy as np
import os
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import pygame
import cv2
import time

# Paths for alarm sound and owner images folder
ALARM_MP3_PATH = "alarm.mp3"
OWNER_IMAGES_FOLDER = "owner_images/"

# Pre-load the face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Email details
FROM_EMAIL = "patilsandesh2311@gmail.com"
PASSWORD = "stlyksrshxjjvuiy"
POLICE_EMAIL = "bannesachin@gmail.com"

# Initialize pygame for alarm
pygame.mixer.init()

# Function to send email notifications with attached images
def send_email(subject, body, to_email, attachment_paths):
    try:
        msg = MIMEMultipart()
        msg['From'] = FROM_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Attach images
        for attachment_path in attachment_paths:
            with open(attachment_path, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(attachment_path)}')
                msg.attach(part)

        # Send email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(FROM_EMAIL, PASSWORD)
        server.sendmail(FROM_EMAIL, to_email, msg.as_string())
        server.quit()
        print("Email sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")

# Function to check if the detected face matches the owner's face
def is_owner(face_descriptor):
    known_face_descriptors = []

    for image_name in os.listdir(OWNER_IMAGES_FOLDER):
        if image_name.endswith(".jpg") or image_name.endswith(".png"):
            img_path = os.path.join(OWNER_IMAGES_FOLDER, image_name)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                shape = predictor(gray, face)
                descriptor = recognizer.compute_face_descriptor(img, shape)
                known_face_descriptors.append(descriptor)

    for known_face in known_face_descriptors:
        distance = np.linalg.norm(np.array(face_descriptor) - np.array(known_face))
        if distance < 0.6:  # Threshold for face recognition
            return True
    return False

# Function to play the alarm
def play_alarm():
    pygame.mixer.music.load(ALARM_MP3_PATH)
    pygame.mixer.music.play()

# Initialize video capture
cap = cv2.VideoCapture(0)
last_email_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video.")
        break

    # Detect faces
    try:
        faces = detector(frame)
        if len(faces) == 0:
            faces = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if not faces:
            print("No faces detected.")
            continue

        image_paths = []
        for face in faces:
            try:
                shape = predictor(frame, face)
                face_descriptor = recognizer.compute_face_descriptor(frame, shape)

                if is_owner(face_descriptor):
                    print("Owner detected. Skipping alarm and email.")
                    continue

                print("Unauthorized person detected. Triggering alarm and email.")
                # Save the frame with the unauthorized face
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = f"violation_{timestamp}.jpg"
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)

                # Send email with the violation images
                current_time = time.time()
                if current_time - last_email_time > 10:  # Prevent frequent emails
                    send_email(
                        subject="Security Alert: Unauthorized Person Detected",
                        body="An unauthorized person was detected. Please review the attached images.",
                        to_email=POLICE_EMAIL,
                        attachment_paths=image_paths
                    )
                    last_email_time = current_time  # Update last email time

                play_alarm()

            except Exception as e:
                print(f"Error processing face: {e}")
    except Exception as e:
        print(f"Error detecting faces: {e}")
        continue

    # Display the video feed
    cv2.imshow("Security Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
