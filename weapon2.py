import cv2
import numpy as np
import pygame
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
import time

# Load YOLO model and configuration
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()  # Correct method to get layer names
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]  # Get output layers

# Load class labels (assuming you've trained it with custom labels for knives and pistols)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Email configuration
SENDER_EMAIL = "patilsandesh2311@gmail.com"
RECEIVER_EMAIL = "bannesachin@gmail.com"
EMAIL_PASSWORD = "stlyksrshxjjvuiy"  # Use an app password if using Gmail

# Function to send an email
def send_email(subject, body, receiver_email, image_path=None):
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = receiver_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    if image_path:
        with open(image_path, 'rb') as f:
            img_data = f.read()
        image = MIMEImage(img_data, name="detected_weapon.jpg")
        msg.attach(image)

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        server.sendmail(SENDER_EMAIL, receiver_email, msg.as_string())
        server.close()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

# Function to play alarm sound
def play_alarm():
    pygame.mixer.init()
    pygame.mixer.music.load("alarm.mp3")  # Make sure this is the correct path to your MP3 file
    pygame.mixer.music.play()

# Function to perform detection
def detect_weapon(img):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # You can adjust the confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression to eliminate redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    return boxes, confidences, class_ids, indexes

# Function to display the results
def display_results(img, boxes, confidences, class_ids, indexes):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])  # Label is the object name, like knife or pistol
            confidence = str(round(confidences[i], 2))

            # Check if the label is "knife" and proceed only if it is
            if label == "knife":  # Only for knife
                color = (0, 255, 0)  # Green for knife
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, f"{label} {confidence}", (x, y - 10), font, 0.6, color, 2)

                # If a knife is detected, trigger the alarm and send an email
                print("Knife detected!")
                play_alarm()

                # Save the frame as an image to send as attachment
                image_path = "detected_knife.jpg"
                cv2.imwrite(image_path, img)

                # Send email with the image attachment
                send_email(
                    "Knife Detected!",
                    "A knife has been detected in the video feed.",
                    RECEIVER_EMAIL,
                    image_path
                )

# Main function for real-time weapon detection
def real_time_detection():
    cap = cv2.VideoCapture(0)  # Use the webcam for live video input
    last_email_time = time.time()

    detection_started = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Wait for user input to start/stop detection
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):  # Press 's' to start detection
            detection_started = True
            print("Detection Started!")

        if key == ord('e'):  # Press 'e' to stop detection
            detection_started = False
            print("Detection Stopped!")

        # Only perform detection if it's started
        if detection_started:
            # Detect weapons in the frame
            boxes, confidences, class_ids, indexes = detect_weapon(frame)

            # Display the results on the frame
            display_results(frame, boxes, confidences, class_ids, indexes)

        # Display the frame
        cv2.imshow("Weapon Detection", frame)

        # Exit loop when 'q' is pressed
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Call the real-time detection function
real_time_detection()
