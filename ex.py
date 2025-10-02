import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox, Entry, Label, Button, Frame
import threading
import time
from datetime import datetime
from PIL import Image, ImageTk

# Global variables
running = False
cap = None
door_locked = True
recognized_faces = set()
last_unlock_time = 0
labels = {}
face_data_path = "face_data"
confidence_threshold = 60
lock_cooldown = 5
face_id_counter = 1

# Ensure face recognizer is available
if not hasattr(cv2, 'face'):
    messagebox.showerror("Error", "cv2.face module not found. Install opencv-contrib-python.")
    raise SystemExit("cv2.face module not available")

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# GUI elements to be defined later
lock_img = None
unlock_img = None
lock_photo = None
unlock_photo = None

# Load labels
def load_labels():
    global face_id_counter
    if os.path.exists("labels.txt"):
        with open("labels.txt", "r") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) == 2:
                  face_id, name = parts
                  labels[int(face_id)] = name
                  face_id_counter = max(face_id_counter, int(face_id) + 1)

def save_labels():
    with open("labels.txt", "w") as f:
        for face_id, name in labels.items():
            f.write(f"{face_id}:{name}\n")

def control_door(state):
    global door_locked
    door_locked = state
    status = "Locked" if door_locked else "Unlocked"
    status_label.config(text=f"Door Status: {status}")
    door_img_label.config(image=lock_photo if door_locked else unlock_photo)
    print(f"Door {status} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def capture_face_data():
    global face_id_counter
    name = name_entry.get().strip()
    if not name:
        messagebox.showerror("Error", "Please enter a valid name")
        return
    if name in labels.values():
        messagebox.showerror("Error", "Name already exists. Use a unique name")
        return

    face_id = face_id_counter
    labels[face_id] = name
    save_labels()
    face_id_counter += 1

    os.makedirs(f"{face_data_path}/{face_id}", exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open camera")
        return

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(f"{face_data_path}/{face_id}/User.{count}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Sample {count}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.imshow("Capturing Face Data", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 500:
            break
    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Info", f"Captured {count} images for {name} (Face ID: {face_id})")

def train_model():
    faces, ids = [], []
    for face_id in os.listdir(face_data_path):
        face_dir = os.path.join(face_data_path, face_id)
        if not os.path.isdir(face_dir):
            continue
        for img in os.listdir(face_dir):
            img_path = os.path.join(face_dir, img)
            gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray_img is None:
                continue
            faces.append(gray_img)
            ids.append(int(face_id))
    if not faces:
        messagebox.showerror("Error", "No training data found")
        return
    recognizer.train(faces, np.array(ids))
    recognizer.save("face_recognizer.yml")
    messagebox.showinfo("Info", "Model training complete!")

def face_recognition_loop():
    global running, cap, door_locked, last_unlock_time
    if not os.path.exists("face_recognizer.yml"):
        messagebox.showerror("Error", "Model not trained. Please train first.")
        running = False
        return
    recognizer.read("face_recognizer.yml")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open camera")
        running = False
        return
    while running:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            face_region = gray[y:y+h, x:x+w]
            face_id, confidence = recognizer.predict(face_region)
            display_conf = 100 - confidence
            name = labels.get(face_id, "Unknown")
            current_time = time.time()
            if display_conf >= confidence_threshold and face_id in labels:
                if current_time - last_unlock_time >= lock_cooldown and door_locked:
                    control_door(False)
                    last_unlock_time = current_time
                    recognized_faces.add(face_id)
                color = (0, 255, 0)
                status_label.config(text=f"Recognized: {name} ({display_conf:.2f}%)")
            else:
                if current_time - last_unlock_time >= lock_cooldown and not door_locked:
                    control_door(True)
                    last_unlock_time = current_time
                name = "Unauthorized"
                color = (0, 0, 255)
                status_label.config(text=f"Unauthorized: {display_conf:.2f}%")
            cv2.putText(frame, f"{name} ({display_conf:.2f}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    if not door_locked:
        control_door(True)

# Start/Stop recognition
def start_recognition():
    global running, recognized_faces
    if not running:
        recognized_faces = set()
        running = True
        start_button.config(state="disabled")
        stop_button.config(state="normal")
        threading.Thread(target=face_recognition_loop, daemon=True).start()

def stop_recognition():
    global running
    running = False
    start_button.config(state="normal")
    stop_button.config(state="disabled")
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    status_label.config(text="Recognition Stopped - Door Locked")
    control_door(True)

def capture_face():
    name_entry.focus()
    capture_face_data()

# Tkinter GUI setup
root = tk.Tk()
root.title("Face Recognition Door Lock System")
root.geometry("650x800")
root.resizable(False, False)
root.configure(bg="#f0f0f0")

# Load lock/unlock images
try:
    lock_img = Image.open(r"D:\pav@\lock door.png")
    unlock_img = Image.open(r"D:\pav@\unlock door.jpg")
    lock_img = lock_img.resize((400, 400), Image.Resampling.LANCZOS)
    unlock_img = unlock_img.resize((400, 400), Image.Resampling.LANCZOS)
    lock_photo = ImageTk.PhotoImage(lock_img)
    unlock_photo = ImageTk.PhotoImage(unlock_img)
except Exception as e:
    messagebox.showerror("Error", f"Image files could not be loaded: {str(e)}")
    root.destroy()
    raise SystemExit

# GUI Layout
frame = Frame(root, bg="#f0f0f0")
frame.pack(pady=20)

Label(frame, text="Enter Name:", font=("Arial", 12), bg="#f0f0f0").pack(pady=5)
name_entry = Entry(frame, font=("Arial", 12), width=20)
name_entry.pack(pady=5)

capture_button = Button(frame, text="Capture Face Data", command=capture_face,
                        font=("Arial", 12), bg="#4CAF50", fg="white", width=20)
capture_button.pack(pady=5)

train_button = Button(frame, text="Train Model", command=train_model,
                      font=("Arial", 12), bg="#2196F3", fg="white", width=20)
train_button.pack(pady=5)

start_button = Button(frame, text="Start Recognition", command=start_recognition,
                      font=("Arial", 12), bg="#FFC107", fg="black", width=20)
start_button.pack(pady=5)

stop_button = Button(frame, text="Stop Recognition", command=stop_recognition,
                     font=("Arial", 12), bg="#F44336", fg="white", width=20, state="disabled")
stop_button.pack(pady=5)

door_img_label = Label(frame, image=lock_photo, bg="#f0f0f0")
door_img_label.pack(pady=10)

status_label = Label(frame, text="Door Status: Locked", font=("Arial", 12), bg="#f0f0f0")
status_label.pack()

# Load previous labels
load_labels()

# Run GUI
root.mainloop()