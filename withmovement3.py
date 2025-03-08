import cv2
import numpy as np
import smtplib
import pygame
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os

# Paths for alarm sound
ALARM_MP3_PATH = "alarm.mp3"

FROM_EMAIL = "patilsandesh2311@gmail.com"
PASSWORD = "stlyksrshxjjvuiy"
TO_EMAIL = "bannesachin@gmail.com"

# Initialize pygame for alarm
pygame.mixer.init()

# Initialize video capture
cap = cv2.VideoCapture(0)

# Read the first frame to initialize the background
ret, first_frame = cap.read()
if not ret:
    print("Failed to capture video.")
    exit()

# Convert the first frame to grayscale
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_gray = cv2.GaussianBlur(first_gray, (21, 21), 0)

# Function to send email notifications
def send_email(subject, body, to_email, attachment_paths=[]):
    try:
        msg = MIMEMultipart()
        msg['From'] = FROM_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Attach images if any
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

# Function to play the alarm sound
def play_alarm():
    pygame.mixer.music.load(ALARM_MP3_PATH)
    pygame.mixer.music.play()

# Movement detection threshold
MOTION_THRESHOLD_AREA = 5000  # Minimum area of movement to consider significant (you can adjust this)

motion_tracking = False  # Flag to track motion detection state

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Check if motion tracking is active
    if motion_tracking:
        # Compute the absolute difference between the first frame and the current frame
        frame_diff = cv2.absdiff(first_gray, gray)

        # Threshold the difference image to find significant changes (movement)
        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

        # Dilate the threshold image to fill in the holes
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                continue  # Ignore small contours

            # Get the bounding box for the detected movement
            (x, y, w, h) = cv2.boundingRect(contour)

            # Check if the detected area is significant enough to trigger the alarm
            if cv2.contourArea(contour) > MOTION_THRESHOLD_AREA:
                motion_detected = True
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                print("Significant movement detected, triggering alarm and email.")

                # Save the frame with movement as an image
                movement_image_path = "movement_detected.jpg"
                cv2.imwrite(movement_image_path, frame)

                # Send email notification
                subject = "Movement Detected"
                body = "Significant movement has been detected. Please check the attached image."
                send_email(subject, body, TO_EMAIL, [movement_image_path])

                # Play alarm
                play_alarm()

        # Set the current frame as the previous frame for the next iteration
        first_gray = gray

    # Display the frame with detected movement
    cv2.imshow("Movement Detection", frame)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):  # Press 's' to start motion detection
        motion_tracking = True
        print("Motion detection started.")

    elif key == ord('x'):  # Press 'x' to stop motion detection
        motion_tracking = False
        print("Motion detection stopped.")

    elif key == ord('q'):  # Press 'q' to quit the program
        break

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()
