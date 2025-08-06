import cv2
import os
import numpy as np
import sqlite3
import tkinter as tk
from tkinter import messagebox, ttk
import time
from picamera2 import Picamera2
import threading
from PIL import Image, ImageTk
import pygame
from libcamera import Transform
import RPi.GPIO as GPIO
import paho.mqtt.client as mqtt
import json
import subprocess
import wave
import struct

# --- Printer imports ---
import usb.core
import usb.util

# === CONFIGURATION ===
SAMPLES_PER_USER = 15
DB_PATH = "faces.db"
FACE_SIZE = (150, 150)
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 480

# DNN Model paths
DNN_PROTO = "models/opencv_face_detector.pbtxt"
DNN_MODEL = "models/opencv_face_detector_uint8.pb"

# MQTT Configuration
MQTT_BROKER = "192.168.255.207"
MQTT_PORT = 1883
MQTT_CLIENT_ID = "barbot_face_recognition"
MQTT_TOPIC_FACE_RECOGNITION = "barbot/face/recognition"
MQTT_TOPIC_DRINK_SELECTION = "drink/selection"
MQTT_TOPIC_MOTOR_STATUS = "motor/status"
MQTT_TOPIC_STIR_RESPONSE = "user/stir_response"
MQTT_TOPIC_DISTANCE = "sensor/distance"
MQTT_TOPIC_CUP_STATUS = "barbot/sensor/capacitive"
MQTT_TOPIC_IR_SENSORS = "sensor/ir"
MQTT_TOPIC_LIQUID_STATUS = "sensor/liquid"
MQTT_TOPIC_LIMIT_SWITCH = "sensor/limit"

# Hardware Pins
TRIG_PIN = 23
ECHO_PIN = 24
IR_GRAPE_PIN = 22
IR_SPRITE_PIN = 7
IR_ORANGE_PIN = 17
LIMIT_SWITCH_PIN = 25
CAPACITIVE_PIN = 12
TRIGGER_DISTANCE = 10.0

# System State Constants
SYSTEM_READY = True
SYSTEM_BUSY = False

# Drink Options
DRINK_OPTIONS = {
    "grape_fizz": "🍇 Grape Fizz 🍇",
    "orange_fizz": "🍊 Orange Fizz 🍊", 
    "purple_sunset": "🍇 Purple Sunset 🍊",
    "orange_on_top": "🧡 Orange on Top",
    "grape_on_top": "💜 Grape on Top"
}

# Global Variables
face_net = None
should_exit = False
root_window = None
mqtt_client = None
current_drink = None
recognized_user = None
system_ready = True
waiting_for_stir_response = False
waiting_for_stir_completion = False
waiting_for_cup = False
sprite_available = True
cup_detected = False
last_liquid_state = None
picam = None

# === PRINTER FUNCTIONS ===
def find_goojprt_printer():
    """Find the GoojPRT printer (now shows as STMicroelectronics 58Printer)"""
    device = usb.core.find(idVendor=0x0483, idProduct=0x5840)
    if device is None:
        device = usb.core.find(bDeviceClass=7)  # Printer class
    return device

def send_to_printer(device, data):
    """Send data to the thermal printer"""
    try:
        if device.is_kernel_driver_active(0):
            try:
                device.detach_kernel_driver(0)
            except usb.core.USBError:
                pass
        device.set_configuration()
        cfg = device.get_active_configuration()
        interface = cfg[(0, 0)]
        ep_out = None
        for ep in interface:
            if usb.util.endpoint_direction(ep.bEndpointAddress) == usb.util.ENDPOINT_OUT:
                ep_out = ep
                break
        if ep_out is None:
            print("Could not find OUT endpoint")
            return False
        chunk_size = 64
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            ep_out.write(chunk)
            time.sleep(0.01)
        return True
    except usb.core.USBError as e:
        print(f"USB Error: {e}")
        return False
    except Exception as e:
        print(f"Error sending to printer: {e}")
        return False

def print_receipt(name, drink):
    """Print a receipt with the customer's name and drink"""
    device = find_goojprt_printer()
    if device is None:
        print("❌ No GoojPRT printer found!")
        return False

    commands = {
        'init': b'\x1B\x40',
        'cut': b'\x1D\x56\x42\x00',
        'feed': b'\n',
        'bold_on': b'\x1B\x45\x01',
        'bold_off': b'\x1B\x45\x00',
        'center': b'\x1B\x61\x01',
        'left': b'\x1B\x61\x00',
        'double_size': b'\x1D\x21\x11',
        'normal_size': b'\x1D\x21\x00'
    }
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    drink_str = drink if drink else "Unknown Drink"
    name_str = name if name else "Customer"
    data = (
        commands['init'] +
        commands['center'] +
        commands['double_size'] +
        b'BARBOT RECEIPT' + commands['feed'] +
        commands['normal_size'] +
        b'================' + commands['feed'] +
        commands['left'] +
        b'Name: ' + name_str.encode() + commands['feed'] +
        b'Drink: ' + drink_str.encode() + commands['feed'] +
        b'Time: ' + now.encode() + commands['feed'] +
        commands['feed'] +
        commands['center'] +
        b'I hope you enjoy your drink <3' + commands['feed'] * 2 +
        commands['cut']
    )
    print("🖨️ Printing receipt...")
    return send_to_printer(device, data)

# === INITIALIZATION ===
def init_all_systems():
    """Initialize all systems: GPIO, MQTT, Database, Camera"""
    global face_net, mqtt_client, picam
    
    print("Initializing BARBOT systems...")
    
    # Initialize GPIO
    GPIO.cleanup()
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(TRIG_PIN, GPIO.OUT)
    GPIO.setup(ECHO_PIN, GPIO.IN)
    GPIO.setup(IR_GRAPE_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(IR_SPRITE_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(IR_ORANGE_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(LIMIT_SWITCH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(CAPACITIVE_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    
    # Initialize pygame
    pygame.init()
    pygame.mouse.set_visible(False)
    
    # Load DNN face detector
    try:
        face_net = cv2.dnn.readNetFromTensorflow(DNN_MODEL, DNN_PROTO)
        print("✅ Face detection model loaded")
    except Exception as e:
        print(f"❌ Failed to load face detection model: {e}")
    
    # Initialize database
    init_db()
    print("✅ Database initialized")
    
    # Initialize camera
    try:
        picam = init_camera()
        print("✅ Camera initialized")
    except Exception as e:
        print(f"❌ Camera initialization failed: {e}")
    
    # Initialize MQTT
    setup_mqtt()
    
    # Start hardware monitoring
    hardware_thread = threading.Thread(target=monitor_hardware, daemon=True)
    hardware_thread.start()
    print("✅ Hardware monitoring started")
    
    print("🚀 All systems initialized successfully!")


def measure_distance():
    GPIO.output(TRIG_PIN, False)
    time.sleep(0.000002)
    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, False)
    
    pulse_start = time.time()
    pulse_end = time.time()
    
    timeout = time.time() + 0.04
    while GPIO.input(ECHO_PIN) == 0 and time.time() < timeout:
        pulse_start = time.time()
    
    timeout = time.time() + 0.04
    while GPIO.input(ECHO_PIN) == 1 and time.time() < timeout:
        pulse_end = time.time()
    
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    distance = round(distance, 2)
    
    return max(2, min(9, distance))

def check_capacitive_sensor():
    """Check capacitive sensor state"""
    return GPIO.input(CAPACITIVE_PIN)

def display_menu():
    """Display drink menu to console"""
    print("\n===== BARBOT MENU =====")
    for i, (drink_id, drink_name) in enumerate(DRINK_OPTIONS.items(), 1):
        print(f"{i}. {drink_name}")
    print("=======================")

def send_drink_selection(drink_id):
    """Send drink selection via MQTT"""
    global current_drink, system_ready
    
    if drink_id in DRINK_OPTIONS:
        # Check if drink requires Sprite and if it's available
        if ("fizz" in drink_id or "sunset" in drink_id) and not sprite_available:
            print("Cannot prepare this drink - Sprite container is empty!")
            return False
            
        current_drink = drink_id
        system_ready = False
        
        # Send to ESP32
        if mqtt_client:
            mqtt_client.publish(MQTT_TOPIC_DRINK_SELECTION, drink_id)
        print(f"\nSelected: {DRINK_OPTIONS[drink_id]}")
        print("Waiting for user approach...")
        return True
    else:
        print("Invalid drink selection")
        return False

def check_liquid_level():
    """Check liquid level"""
    global sprite_available, last_liquid_state
    
    # Read capacitive sensor (inverted logic)
    liquid_detected = not GPIO.input(CAPACITIVE_PIN)
    
    # Only send message if state changed
    if liquid_detected != last_liquid_state:
        if liquid_detected:
            sprite_available = False
            if mqtt_client:
                mqtt_client.publish(MQTT_TOPIC_LIQUID_STATUS, "sprite_empty")
            print("Warning: Sprite container is empty!")
        else:
            sprite_available = True
            if mqtt_client:
                mqtt_client.publish(MQTT_TOPIC_LIQUID_STATUS, "sprite_available")
            print("Sprite container has liquid")
        
        last_liquid_state = liquid_detected

def monitor_ir_sensors():
    """Monitor IR sensors and send continuous activation messages while active"""
    # Track the state of each sensor
    sensor_states = {
        'grape': {
            'pin': IR_GRAPE_PIN,
            'last_state': None,
            'last_message_time': 0
        },
        'sprite': {
            'pin': IR_SPRITE_PIN,
            'last_state': None,
            'last_message_time': 0
        },
        'orange': {
            'pin': IR_ORANGE_PIN,
            'last_state': None,
            'last_message_time': 0
        }
    }
    
    # Time between repeated messages while sensor stays active (in seconds)
    MESSAGE_INTERVAL = 7.0  # Send confirmation every 1 second while active

    while not should_exit:
        try:
            current_time = time.time()
            
            for sensor_name, sensor_data in sensor_states.items():
                current_state = not GPIO.input(sensor_data['pin'])  # Inverted because IR sensors are active-low
                
                if current_state:  # Sensor is currently active
                    # Send message if it's time for another update
                    if (current_time - sensor_data['last_message_time']) >= MESSAGE_INTERVAL:
                        if mqtt_client:
                            mqtt_client.publish(MQTT_TOPIC_IR_SENSORS, sensor_name)
                        print(f"IR: {sensor_name.capitalize()} position confirmed")
                        sensor_data['last_message_time'] = current_time
                
                # Update last state regardless
                sensor_data['last_state'] = current_state
            
            time.sleep(0.1)  # Short delay to prevent CPU overuse
            
        except Exception as e:
            print(f"IR monitoring error: {e}")
            time.sleep(1)  # Longer delay if error occurs
def monitor_limit_switch():
    print("Limit switch monitor starting...")
    last_state = None
    while True:
        try:
            # Wait for MQTT connection
            while not mqtt_client.is_connected():
                time.sleep(0.5)
                
            # Read switch state (inverted because using pull-up)
            current_state = not GPIO.input(LIMIT_SWITCH_PIN)
            
            if current_state != last_state:
                print(f"Limit switch state changed to: {'TRIGGERED' if current_state else 'RELEASED'}")
                
                # Publish with QoS 1 to ensure delivery
                try:
                    if current_state:
                        mqtt_client.publish(MQTT_TOPIC_LIMIT_SWITCH, "danger", qos=1)
                        # Trigger system reset when limit switch is activated
                        handle_limit_switch_trigger()
                    else:
                        mqtt_client.publish(MQTT_TOPIC_LIMIT_SWITCH, "safe", qos=1)
                except Exception as e:
                    print(f"Error publishing limit state: {e}")
                
                last_state = current_state
            
            time.sleep(0.05)  # Faster polling
            
        except Exception as e:
            print(f"Error in limit switch monitor: {e}")
            time.sleep(1)

def handle_limit_switch_trigger():
    """Handle limit switch trigger by resetting system to face recognition"""
    global should_exit, current_drink, recognized_user, system_ready, waiting_for_stir_response, waiting_for_cup, root_window
    
    print("🚨 LIMIT SWITCH TRIGGERED - Resetting system...")
    
    # Reset all system states
    current_drink = None
    recognized_user = None
    system_ready = True
    waiting_for_stir_response = False
    waiting_for_cup = False
    
    # Stop any ongoing drink preparation
    if mqtt_client:
        mqtt_client.publish(MQTT_TOPIC_MOTOR_STATUS, "stop")
        print("🛑 Stopping drink preparation due to limit switch")
    
    # Show emergency reset screen
    if root_window:
        root_window.after(0, show_emergency_reset_screen)

def show_emergency_reset_screen():
    """Show emergency reset screen when limit switch is triggered"""
    global should_exit
    
    def on_continue():
        root.quit()
    
    root = get_root_window()
    clear_window(root)
    
    main_frame = tk.Frame(root, bg="#f5f1ed")
    main_frame.pack(fill="both", expand=True, padx=20, pady=20)
    
    # Emergency message
    emergency_frame = tk.Frame(main_frame, bg="#f5f1ed")
    emergency_frame.pack(expand=True, fill="both")
    
    icon_label = tk.Label(emergency_frame, text="🚨", 
                         font=("Helvetica", 72), bg="#f5f1ed")
    icon_label.pack(pady=20)
    
    title_label = tk.Label(emergency_frame, text="Safety Reset",
                          font=("Helvetica", 32, "bold"),
                          bg="#f5f1ed", fg="#ff6b6b")
    title_label.pack(pady=(0, 10))
    
    message_label = tk.Label(emergency_frame, text="System has been reset for safety.\nReturning to main menu...",
                            font=("Helvetica", 20),
                            bg="#f5f1ed", fg="#666666",
                            justify="center")
    message_label.pack(pady=(0, 30))
    
    status_label = tk.Label(emergency_frame, text="All operations have been stopped",
                           font=("Helvetica", 16),
                           bg="#f5f1ed", fg="#ffa726")
    status_label.pack(pady=10)
    
    # Continue button
    button_frame = tk.Frame(main_frame, bg="#f5f1ed")
    button_frame.pack(fill="x", pady=20)
    
    TouchButton(button_frame, "CONTINUE", on_continue, 
                width=250, height=70, font_size=20, bg_color="#ff7043").pack(anchor="center")
    
    root.after(100, lambda: root.focus_set())
    
    # Auto-continue after 3 seconds
    root.after(3000, on_continue)
    
    try:
        root.mainloop()
    except:
        pass

def monitor_hardware():
    """Monitor all hardware sensors in background thread"""
    global cup_detected
    
    last_cup_state = False
    
    # Start dedicated monitoring threads
    ir_thread = threading.Thread(target=monitor_ir_sensors, daemon=True)
    ir_thread.start()
    
    limit_thread = threading.Thread(target=monitor_limit_switch, daemon=True)
    limit_thread.start()
    
    while not should_exit:
        try:
            # Check liquid level periodically
            check_liquid_level()
            
            # Measure distance
            dist = measure_distance()
            print(f"DEBUG: Distance = {dist}cm, Trigger = {TRIGGER_DISTANCE}cm")
            if mqtt_client:
                mqtt_client.publish(MQTT_TOPIC_DISTANCE, str(dist))
            
            # Cup detection logic
            cup_detected = dist <= TRIGGER_DISTANCE
            print(f"DEBUG: Cup detected = {cup_detected}, Last state = {last_cup_state}")
            
            # Send cup detection status
            if cup_detected != last_cup_state:
                if cup_detected:
                    print(f"DEBUG: Cup state changed to {cup_detected}")
                    if mqtt_client:
                        mqtt_client.publish(MQTT_TOPIC_CUP_STATUS, "cup_detected")
                    print(f"Cup detected at {dist}cm")
                else:
                    if mqtt_client:
                        mqtt_client.publish(MQTT_TOPIC_CUP_STATUS, "cup_removed")
                        
                    print(f"Cup removed - distance {dist}cm")
                last_cup_state = cup_detected
            
            time.sleep(0.5)
        except Exception as e:
            print(f"Hardware monitoring error: {e}")
            time.sleep(1)

def init_db():
    """Initialize face recognition database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            drink TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS face_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            sample_data BLOB,
            face_features BLOB,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()

def init_camera():
    """Initialize camera with optimal settings"""
    picam = Picamera2()
    config = picam.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"},
        transform=Transform(hflip=False, vflip=False)
    )
    picam.configure(config)
    picam.set_controls({
        "AeEnable": True,
        "AwbEnable": True,
        "Sharpness": 1.0,
        "Contrast": 1.0,
        "Brightness": 0.0,
        "NoiseReductionMode": 2
    })
    picam.start()
    time.sleep(2)
    return picam

def setup_mqtt():
    """Setup MQTT client and connections"""
    global mqtt_client
    
    mqtt_client = mqtt.Client(client_id=MQTT_CLIENT_ID)
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message
    mqtt_client.on_disconnect = on_mqtt_disconnect
    
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        print(f"✅ MQTT connected to {MQTT_BROKER}:{MQTT_PORT}")
    except Exception as e:
        print(f"❌ MQTT connection failed: {e}")

def on_mqtt_connect(client, userdata, flags, rc):
    """MQTT connection callback"""
    print(f"MQTT Connected with code {rc}")
    client.subscribe(MQTT_TOPIC_MOTOR_STATUS)
    client.subscribe(MQTT_TOPIC_CUP_STATUS)
    client.subscribe(MQTT_TOPIC_LIQUID_STATUS)

def on_mqtt_disconnect(client, userdata, rc):
    """MQTT disconnection callback"""
    print(f"MQTT Disconnected with code {rc}")

def on_mqtt_message(client, userdata, msg):
    """Handle incoming MQTT messages"""
    global system_ready, waiting_for_stir_response, current_drink, waiting_for_cup
    
    topic = msg.topic
    message = msg.payload.decode()
    
    print(f"📨 MQTT: {topic} -> {message}")
    
    if topic == MQTT_TOPIC_MOTOR_STATUS:
        if "complete. Would you like to stir" in message:
            waiting_for_stir_response = True
            system_ready = False
            # Ensure we're in the main thread for UI operations
            if root_window:
                root_window.after(0, show_stir_question_mqtt)
                
        elif "stirring complete" in message.lower():
            waiting_for_stir_response = False
            system_ready = True
            current_drink = None
            if root_window:
                root_window.after(0, show_drink_ready_final)
                
        elif "complete" in message.lower() and "stir" not in message.lower():
            if not waiting_for_stir_response:
                system_ready = True
                if root_window:
                    root_window.after(0, show_drink_ready_screen)
                # Print receipt here
                print_receipt(name or "Customer", DRINK_OPTIONS.get(current_drink, "Unknown Drink"))
    
    elif topic == MQTT_TOPIC_CUP_STATUS:
        print(f"DEBUG: MQTT_TOPIC_CUP_STATUS received, message={message}, current_drink={current_drink}, waiting_for_cup={waiting_for_cup}")
        if message == "cup_detected" and current_drink and waiting_for_cup:
            waiting_for_cup = True
            print(f"🥤 Cup detected! Starting {current_drink} preparation")
            start_drink_preparation()

# === FACE RECOGNITION FUNCTIONS ===
def detect_faces_dnn(frame):
    """Detect faces using DNN model"""
    global face_net
    if face_net is None:
        return []
        
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
    face_net.setInput(blob)
    detections = face_net.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2-x1, y2-y1, confidence))
    
    return faces

def extract_face_features(face_img):
    """Extract face features for recognition"""
    try:
        if face_img is None or face_img.size == 0:
            return None
            
        face_resized = cv2.resize(face_img, FACE_SIZE)
        face_eq = cv2.equalizeHist(face_resized)
        face_blur = cv2.GaussianBlur(face_eq, (3, 3), 0)
        
        h, w = face_blur.shape
        cell_h, cell_w = h // 4, w // 4
        
        regional_features = []
        for i in range(4):
            for j in range(4):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cell = face_blur[y1:y2, x1:x2]
                
                hist = cv2.calcHist([cell], [0], None, [32], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                regional_features.extend(hist)
        
        regional_features = np.array(regional_features)
        
        grad_x = cv2.Sobel(face_blur, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(face_blur, cv2.CV_64F, 0, 1, ksize=3)
        
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        mag_hist = cv2.calcHist([grad_mag.astype(np.uint8)], [0], None, [32], [0, 256])
        mag_hist = cv2.normalize(mag_hist, mag_hist).flatten()
        
        grad_orient = np.arctan2(grad_y, grad_x) * 180 / np.pi
        grad_orient = (grad_orient + 180) % 360
        orient_hist = np.histogram(grad_orient.flatten(), bins=36, range=(0, 360))[0]
        orient_hist = orient_hist / (np.sum(orient_hist) + 1e-7)
        
        edges = cv2.Canny(face_blur, 50, 150)
        edge_density = []
        for i in range(4):
            for j in range(4):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cell_edges = edges[y1:y2, x1:x2]
                density = np.sum(cell_edges > 0) / (cell_h * cell_w)
                edge_density.append(density)
        
        edge_density = np.array(edge_density)
        
        combined_features = np.concatenate([
            regional_features,
            mag_hist,
            orient_hist,
            edge_density
        ])
        
        combined_features = combined_features / (np.linalg.norm(combined_features) + 1e-7)
        
        return combined_features.astype(np.float32)
        
    except Exception as e:
        print(f"Error extracting face features: {e}")
        return None

def recognize_face(face_features):
    """Recognize face by comparing with database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute("""
            SELECT fs.face_features, u.name, u.drink 
            FROM face_samples fs 
            JOIN users u ON fs.user_id = u.id
        """)
        
        results = c.fetchall()
        conn.close()
        
        if not results or face_features is None:
            return None, None
        
        user_scores = {}
        
        for stored_features_blob, name, drink in results:
            stored_features = np.frombuffer(stored_features_blob, dtype=np.float32)
            distance = np.linalg.norm(face_features - stored_features)
            
            if name not in user_scores:
                user_scores[name] = {'distances': [], 'drink': drink}
            user_scores[name]['distances'].append(distance)
        
        best_user = None
        best_avg_distance = float('inf')
        
        for name, data in user_scores.items():
            avg_distance = np.mean(data['distances'])
            min_distance = min(data['distances'])
            combined_score = (avg_distance * 0.7) + (min_distance * 0.3)
            
            if combined_score < best_avg_distance:
                best_avg_distance = combined_score
                best_user = (name, data['drink'])
        
        if best_avg_distance < 0.8:
            print(f"✅ Face recognized: {best_user[0]} (confidence: {1-best_avg_distance:.2f})")
            return best_user
        else:
            print(f"❌ Face not recognized (best score: {best_avg_distance:.3f})")
            return None, None
            
    except Exception as e:
        print(f"Recognition error: {e}")
        return None, None

def save_user_to_db(name, drink, face_features_list):
    """Save user and face features to database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute("INSERT INTO users (name, drink) VALUES (?, ?)", (name, drink))
        user_id = c.lastrowid
        
        for features in face_features_list:
            if features is not None:
                features_blob = features.tobytes()
                c.execute("INSERT INTO face_samples (user_id, face_features) VALUES (?, ?)",
                         (user_id, features_blob))
        
        conn.commit()
        conn.close()
        print(f"✅ Saved {len(face_features_list)} face samples for {name}")
        return True
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

# === UI COMPONENTS ===
class TouchButton(tk.Canvas):
    def __init__(self, parent, text, command, width=180, height=60, bg_color="#f4a6c1", 
                 text_color="white", font_size=18, corner_radius=15):
        super().__init__(parent, width=width, height=height, highlightthickness=0, bd=0)
        self.command = command
        self.bg_color = bg_color
        self.text_color = text_color
        self.corner_radius = corner_radius
        
        self.create_rounded_rect(0, 0, width, height, radius=corner_radius, fill=bg_color)
        self.create_text(width//2, height//2, text=text, fill=text_color, 
                        font=("Helvetica", font_size, "bold"))
        
        self.bind("<Button-1>", self.on_press)
        self.bind("<ButtonRelease-1>", self.on_release)
        
    def create_rounded_rect(self, x1, y1, x2, y2, radius=15, **kwargs):
        points = []
        for i in range(0, 91, 15):
            angle = i * np.pi / 180
            points.extend([x1 + radius - radius * np.cos(angle), y1 + radius - radius * np.sin(angle)])
        for i in range(90, 181, 15):
            angle = i * np.pi / 180
            points.extend([x2 - radius - radius * np.cos(angle), y1 + radius - radius * np.sin(angle)])
        for i in range(180, 271, 15):
            angle = i * np.pi / 180
            points.extend([x2 - radius - radius * np.cos(angle), y2 - radius - radius * np.sin(angle)])
        for i in range(270, 361, 15):
            angle = i * np.pi / 180
            points.extend([x1 + radius - radius * np.cos(angle), y2 - radius - radius * np.sin(angle)])
        
        return self.create_polygon(points, **kwargs, smooth=True)
        
    def on_press(self, event):
        self.itemconfig(1, fill="#d48fb1")
        
    def on_release(self, event):
        self.itemconfig(1, fill=self.bg_color)
        if self.command:
            self.command()

class TouchEntry(tk.Entry):
    def __init__(self, parent, *args, **kwargs):
        kwargs.setdefault('font', ('Helvetica', 16))
        kwargs.setdefault('bd', 3)
        kwargs.setdefault('relief', 'solid')
        kwargs.setdefault('bg', '#ffffff')
        kwargs.setdefault('fg', '#333333')
        kwargs.setdefault('insertbackground', '#f4a6c1')
        kwargs.setdefault('justify', 'left')
        kwargs.setdefault('width', 30)
        
        super().__init__(parent, *args, **kwargs)
        
        self.bind("<Button-1>", self.on_click)
        self.bind("<FocusIn>", self.on_focus_in)
        self.bind("<FocusOut>", self.on_focus_out)
        
    def on_click(self, event):
        self.focus_set()
        self.selection_clear()
        self.icursor(tk.END)
        
    def on_focus_in(self, event):
        self.config(bg='#f0f8ff', bd=4)
        
    def on_focus_out(self, event):
        self.config(bg='#ffffff', bd=3)

# === WINDOW MANAGEMENT ===
def get_root_window():
    """Get or create the single root window"""
    global root_window
    if root_window is None or not root_window.winfo_exists():
        root_window = tk.Tk()
        root_window.geometry(f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}")
        root_window.attributes('-fullscreen', True)
        root_window.config(bg="#f5f1ed")
        root_window.focus_set()
        
        def handle_escape(event=None):
            global should_exit
            print("ESC pressed - exiting BARBOT...")
            should_exit = True
            root_window.quit()
        
        root_window.bind("<KeyPress-Escape>", handle_escape)
        root_window.bind("<Escape>", handle_escape)
        root_window.bind_all("<KeyPress-Escape>", handle_escape)
        
        root_window.attributes('-topmost', True)
        root_window.after(100, lambda: root_window.attributes('-topmost', False))
    
    return root_window

def clear_window(root):
    """Clear all widgets from window"""
    for widget in root.winfo_children():
        widget.destroy()

# === CAMERA PREVIEW ===
class CameraPreview(tk.Toplevel):
    def __init__(self, parent, title="Camera Preview", duration=5, capture_mode=False):
        super().__init__(parent)
        
        self.title_text = title
        self.duration = duration
        self.capture_mode = capture_mode
        self.running = False
        self.captured_face = None
        self.face_detected = False
        
        self.geometry(f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}")
        self.attributes('-fullscreen', True)
        self.config(bg="#f5f1ed")
        
        self.focus_set()
        self.bind("<KeyPress-Escape>", self.handle_escape)
        
        main_frame = tk.Frame(self, bg="#f5f1ed")
        main_frame.pack(fill="both", expand=True, padx=15, pady=10)
        
        title_label = tk.Label(main_frame, text=self.title_text,
                             font=("Helvetica", 20, "bold"),
                             bg="#f5f1ed", fg="#f4a6c1")
        title_label.pack(pady=(5, 10))
        
        camera_container = tk.Frame(main_frame, bg="#f5f1ed")
        camera_container.pack(expand=True, fill="both")
        
        self.camera_frame = tk.Frame(camera_container, bg="#333333", relief="solid", bd=3)
        self.camera_frame.pack(anchor="center", padx=10, pady=10)
        
        self.camera_label = tk.Label(self.camera_frame, bg="#333333")
        self.camera_label.pack(padx=8, pady=8)
        
        status_frame = tk.Frame(main_frame, bg="#f5f1ed")
        status_frame.pack(fill="x", pady=(5, 5))
        
        self.status_label = tk.Label(status_frame, text="Looking for face...",
                                   font=("Helvetica", 16),
                                   bg="#f5f1ed", fg="#f4a6c1")
        self.status_label.pack()
        
        self.start_time = time.time()
        self.running = True
        self.update_camera_feed()
        
        self.after(self.duration * 1000, self.stop_preview)
    
    def handle_escape(self, event=None):
        global should_exit
        should_exit = True
        self.stop_preview()
    
    def update_camera_feed(self):
        global should_exit, picam
        
        if not self.running or should_exit or picam is None:
            return
            
        try:
            frame = picam.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            
            faces = detect_faces_dnn(frame_bgr)
            
            display_frame = frame.copy()
            face_found = False
            
            for (x, y, w, h, confidence) in faces:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (244, 166, 193), 3)
                conf_text = f"{confidence:.2f}"
                cv2.putText(display_frame, conf_text, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (244, 166, 193), 2)
                
                face_found = True
                
                if self.capture_mode and confidence > 0.7:
                    padding = 10
                    x_pad = max(0, x - padding)
                    y_pad = max(0, y - padding)
                    w_pad = min(gray.shape[1] - x_pad, w + 2*padding)
                    h_pad = min(gray.shape[0] - y_pad, h + 2*padding)
                    
                    face_img = gray[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
                    
                    if face_img.size > 0:
                        self.captured_face = cv2.resize(face_img, FACE_SIZE)
                        self.face_detected = True
            
            if face_found:
                self.status_label.config(text="Face detected! Stay still...", fg="#7dd3c0")
            else:
                self.status_label.config(text="Looking for face...", fg="#f4a6c1")
            
            display_frame = cv2.resize(display_frame, (700, 520))
            
            image = Image.fromarray(display_frame)
            photo = ImageTk.PhotoImage(image)
            self.camera_label.config(image=photo)
            self.camera_label.image = photo
            
        except Exception as e:
            print(f"Camera feed error: {e}")
            self.status_label.config(text="Camera error", fg="red")
        
        if self.running and not should_exit:
            self.after(50, self.update_camera_feed)
    
    def stop_preview(self):
        self.running = False
        try:
            self.destroy()
        except:
            pass
    
    def show(self):
        try:
            self.wait_window()
        except:
            pass
        return self.captured_face if self.capture_mode else None

# === MAIN UI SCREENS ===
def show_welcome_screen():
    """Main welcome screen for BARBOT"""
    global should_exit
    
    def on_face_recognition():
        result[0] = "face_recognition"
        root.quit()
    
    def on_manual_order():
        result[0] = "manual_order"
        root.quit()
    
    def on_register():
        result[0] = "register"
        root.quit()
    
    result = [None]
    
    root = get_root_window()
    clear_window(root)
    root.title("BARBOT - Smart Drink Machine")
    
    main_frame = tk.Frame(root, bg="#f5f1ed")
    main_frame.pack(fill="both", expand=True, padx=20, pady=20)
    
    # Header
    header_frame = tk.Frame(main_frame, bg="#f5f1ed")
    header_frame.pack(fill="x", pady=(10, 20))
    
    title_label = tk.Label(header_frame, text="🤖 BARBOT 🍹",
                         font=("Helvetica", 32, "bold"),
                         bg="#f5f1ed", fg="#f4a6c1")
    title_label.pack()
    
    subtitle_label = tk.Label(header_frame, text="Smart Facial Recognition Drink Machine",
                            font=("Helvetica", 18),
                            bg="#f5f1ed", fg="#666666")
    subtitle_label.pack(pady=(5, 0))
    
    # Status indicators
    status_frame = tk.Frame(main_frame, bg="#f5f1ed")
    status_frame.pack(fill="x", pady=(0, 20))
    
    # System status indicators
    camera_status = "🟢 Camera Ready" if picam else "🔴 Camera Error"
    mqtt_status = "🟢 System Connected" if mqtt_client else "🔴 Connection Error"
    liquid_status = "🟢 Sprite Available" if sprite_available else "🟡 Sprite Low"
    
    status_text = f"{camera_status} | {mqtt_status} | {liquid_status}"
    status_label = tk.Label(status_frame, text=status_text,
                            font=("Helvetica", 12),
                            bg="#f5f1ed", fg="#888888")
    status_label.pack()
    
    # Main options
    options_frame = tk.Frame(main_frame, bg="#f5f1ed")
    options_frame.pack(expand=True, fill="both")
    
    # Face Recognition Option
    face_frame = tk.Frame(options_frame, bg="#f5f1ed")
    face_frame.pack(pady=15)
    
    face_icon = tk.Label(face_frame, text="👤", font=("Helvetica", 48), bg="#f5f1ed")
    face_icon.pack()
    
    TouchButton(face_frame, "FACE RECOGNITION", on_face_recognition, 
                width=280, height=70, font_size=18).pack(pady=10)
    
    face_desc = tk.Label(face_frame, text="Let me recognize you and suggest your favorite drink",
                        font=("Helvetica", 14), bg="#f5f1ed", fg="#666666")
    face_desc.pack()
    
    # Manual Order Option
    manual_frame = tk.Frame(options_frame, bg="#f5f1ed")
    manual_frame.pack(pady=15)
    
    manual_icon = tk.Label(manual_frame, text="🍹", font=("Helvetica", 48), bg="#f5f1ed")
    manual_icon.pack()
    
    TouchButton(manual_frame, "ORDER DRINK", on_manual_order,
                width=280, height=70, font_size=18, bg_color="#7dd3c0").pack(pady=10)
    
    manual_desc = tk.Label(manual_frame, text="Browse our drink menu and order directly",
                            font=("Helvetica", 14), bg="#f5f1ed", fg="#666666")
    manual_desc.pack()
    
    # Register Option
    register_frame = tk.Frame(options_frame, bg="#f5f1ed")
    register_frame.pack(pady=15)
    
    register_icon = tk.Label(register_frame, text="📝", font=("Helvetica", 48), bg="#f5f1ed")
    register_icon.pack()
    
    TouchButton(register_frame, "NEW USER", on_register,
                width=280, height=70, font_size=18, bg_color="#ffa726").pack(pady=10)
    
    register_desc = tk.Label(register_frame, text="Register your face for personalized service",
                            font=("Helvetica", 14), bg="#f5f1ed", fg="#666666")
    register_desc.pack()
    
    root.after(100, lambda: root.focus_set())
    
    try:
        root.mainloop()
    except:
        pass
    
    if should_exit:
        return None
    
    return result[0]

def capture_and_recognize_face():
   """Capture face and attempt recognition"""
   global should_exit, recognized_user
   
   if should_exit:
       return None, None
   
   # Show processing screen
   show_processing_screen("LOOK AT THE CAMERA", "Capturing your face...", 1)
   
   if should_exit:
       return None, None
   
   # Capture face
   root = get_root_window()
   preview = CameraPreview(root, "Look directly at the camera", duration=5, capture_mode=True)
   face = preview.show()
   
   if should_exit:
       return None, None
   
   if face is None:
       show_error_screen("No face detected. Please try again.")
       return None, None
   
   # Show processing screen
   show_processing_screen("ANALYZING...", "Checking if I know you...", 2)
   
   if should_exit:
       return None, None
   
   # Try to recognize
   features = extract_face_features(face)
   if features is not None:
       name, drink = recognize_face(features)
       recognized_user = (name, drink) if name else None
       return name, drink
   
   return None, None

def show_name_input_screen():
   """Registration screen for new users"""
   global should_exit
   result = [None, None]
   
   def on_submit():
       name = name_entry.get().strip()
       drink = drink_entry.get().strip()
       if name and drink:
           result[0] = name
           result[1] = drink
           root.quit()
       else:
           error_label.config(text="Please fill in both fields!", fg="red")
   
   def clear_error():
       error_label.config(text="")
   
   root = get_root_window()
   clear_window(root)
   root.title("New User Registration")
   
   main_frame = tk.Frame(root, bg="#f5f1ed")
   main_frame.pack(fill="both", expand=True, padx=40, pady=30)
   
   title_label = tk.Label(main_frame, text="👋 Welcome to BARBOT!",
                        font=("Helvetica", 28, "bold"),
                        bg="#f5f1ed", fg="#f4a6c1")
   title_label.pack(pady=(10, 20))
   
   subtitle_label = tk.Label(main_frame, text="Let's get you registered for personalized service",
                           font=("Helvetica", 16),
                           bg="#f5f1ed", fg="#666666")
   subtitle_label.pack(pady=(0, 30))
   
   # Name field
   name_label = tk.Label(main_frame, text="Your Name:",
                       font=("Helvetica", 18, "bold"),
                       bg="#f5f1ed", fg="#333333")
   name_label.pack(anchor="w", pady=(0, 5))
   
   name_entry = TouchEntry(main_frame, font=("Helvetica", 18))
   name_entry.pack(fill="x", pady=(0, 20), ipady=12)
   name_entry.bind("<KeyPress>", lambda e: clear_error())
   
   # Drink field
   drink_label = tk.Label(main_frame, text="Favorite Drink:",
                        font=("Helvetica", 18, "bold"),
                        bg="#f5f1ed", fg="#333333")
   drink_label.pack(anchor="w", pady=(0, 5))
   
   drink_entry = TouchEntry(main_frame, font=("Helvetica", 18))
   drink_entry.pack(fill="x", pady=(0, 20), ipady=12)
   drink_entry.bind("<KeyPress>", lambda e: clear_error())
   
   # Error label
   error_label = tk.Label(main_frame, text="",
                        font=("Helvetica", 14),
                        bg="#f5f1ed", fg="red")
   error_label.pack(pady=(0, 20))
   
   # Submit button
   button_frame = tk.Frame(main_frame, bg="#f5f1ed")
   button_frame.pack(fill="x", pady=20)
   
   submit_btn = TouchButton(button_frame, "CONTINUE", on_submit, 
                           width=250, height=70, font_size=20)
   submit_btn.pack(anchor="center")
   
   # Bind Enter key
   name_entry.bind('<Return>', lambda e: on_submit())
   drink_entry.bind('<Return>', lambda e: on_submit())
   
   root.after(100, lambda: name_entry.focus_set())
   
   try:
       root.mainloop()
   except:
       pass
   
   if should_exit:
       return None, None
   
   return result[0], result[1]

def register_new_user(name, drink):
   """Register new user with face samples"""
   global should_exit
   
   if should_exit:
       return False
   
   print(f"Registering {name} with favorite drink: {drink}")
   
   features_list = []
   root = get_root_window()
   
   for i in range(3):  # Capture 3 samples
       if should_exit:
           return False
       
       show_processing_screen(f"SAMPLE {i+1}/3", f"Look directly at the camera, {name}", 1)
       
       if should_exit:
           return False
       
       preview = CameraPreview(root, f"Sample {i+1}/3 - Stay still", duration=4, capture_mode=True)
       face = preview.show()
       
       if should_exit:
           return False
       
       if face is not None:
           if face.shape[0] >= 50 and face.shape[1] >= 50:
               contrast = np.std(face)
               if contrast > 15:
                   features = extract_face_features(face)
                   if features is not None:
                       features_list.append(features)
                       print(f"✅ Sample {i+1} captured successfully")
                   else:
                       print(f"❌ Feature extraction failed for sample {i+1}")
               else:
                   print(f"❌ Sample {i+1} too low contrast")
           else:
               print(f"❌ Sample {i+1} too small")
       else:
           print(f"❌ No face detected in sample {i+1}")
   
   if len(features_list) >= 2:
       success = save_user_to_db(name, drink, features_list)
       if success:
           show_processing_screen("SUCCESS!", f"Welcome to BARBOT, {name}!", 2)
           return True
       else:
           show_error_screen("Registration failed. Please try again.")
           return False
   else:
       show_error_screen(f"Not enough good samples captured ({len(features_list)}/2). Please try again.")
       return False

def show_recognized_user_screen(name, drink):
    """Welcome back screen for recognized users with favorite vs new drink choice"""
    global should_exit
    result = [None]
    
    def on_favorite():
        result[0] = "favorite"
        root.quit()
    
    def on_new_drink():
        result[0] = "new_drink"
        root.quit()
    
    def on_cancel():
        result[0] = "cancel"
        root.quit()
    
    root = get_root_window()
    clear_window(root)
    
    main_frame = tk.Frame(root, bg="#f5f1ed")
    main_frame.pack(fill="both", expand=True, padx=20, pady=20)
    
    # Welcome message
    welcome_frame = tk.Frame(main_frame, bg="#f5f1ed")
    welcome_frame.pack(fill="x", pady=(20, 30))
    
    emoji_label = tk.Label(welcome_frame, text="🎉", 
                          font=("Helvetica", 48), bg="#f5f1ed")
    emoji_label.pack()
    
    title_label = tk.Label(welcome_frame, text=f"Welcome back, {name}!",
                         font=("Helvetica", 28, "bold"),
                         bg="#f5f1ed", fg="#f4a6c1")
    title_label.pack(pady=(10, 5))
    
    drink_label = tk.Label(welcome_frame, text=f"Your favorite: {drink}",
                         font=("Helvetica", 20),
                         bg="#f5f1ed", fg="#7dd3c0")
    drink_label.pack(pady=(0, 20))
    
    # Options
    options_frame = tk.Frame(main_frame, bg="#f5f1ed")
    options_frame.pack(expand=True, fill="both")
    
    question_label = tk.Label(options_frame, text="What would you like today?",
                            font=("Helvetica", 22),
                            bg="#f5f1ed", fg="#333333")
    question_label.pack(pady=(0, 30))
    
    # Buttons - Updated with new option
    button_frame = tk.Frame(options_frame, bg="#f5f1ed")
    button_frame.pack(expand=True)
    
    TouchButton(button_frame, f"MY FAVORITE\n{drink}", on_favorite, 
                width=250, height=100, font_size=16).pack(pady=10)
    
    TouchButton(button_frame, "TRY SOMETHING NEW", on_new_drink, 
                width=250, height=80, font_size=16, bg_color="#7dd3c0").pack(pady=10)
    
    TouchButton(button_frame, "CANCEL", on_cancel, 
                width=250, height=60, font_size=16, bg_color="#ff7043").pack(pady=10)
    
    root.after(100, lambda: root.focus_set())
    
    try:
        root.mainloop()
    except:
        pass
    
    if should_exit:
        return None
    
    return result[0]

def show_drink_menu():
   """Show drink selection menu"""
   global should_exit
   result = [None]
   
   def select_drink(drink_id):
       result[0] = drink_id
       root.quit()
   
   def go_back():
       result[0] = "back"
       root.quit()
   
   root = get_root_window()
   clear_window(root)
   
   main_frame = tk.Frame(root, bg="#f5f1ed")
   main_frame.pack(fill="both", expand=True, padx=20, pady=20)
   
   # Header
   header_frame = tk.Frame(main_frame, bg="#f5f1ed")
   header_frame.pack(fill="x", pady=(10, 20))
   
   title_label = tk.Label(header_frame, text="🍹 Drink Menu",
                        font=("Helvetica", 28, "bold"),
                        bg="#f5f1ed", fg="#f4a6c1")
   title_label.pack()
   
   subtitle_label = tk.Label(header_frame, text="Choose your perfect drink",
                           font=("Helvetica", 16),
                           bg="#f5f1ed", fg="#666666")
   subtitle_label.pack(pady=(5, 0))
   
   # Drinks grid
   drinks_frame = tk.Frame(main_frame, bg="#f5f1ed")
   drinks_frame.pack(expand=True, fill="both", pady=20)
   
   drinks_list = list(DRINK_OPTIONS.items())
   
   # Create 2x3 grid of drinks
   for i, (drink_id, drink_name) in enumerate(drinks_list):
       row = i // 2
       col = i % 2
       
       drink_frame = tk.Frame(drinks_frame, bg="#f5f1ed")
       drink_frame.grid(row=row, column=col, padx=20, pady=15, sticky="ew")
       
       # Check if drink is available
       available = True
       if ("fizz" in drink_id or "sunset" in drink_id) and not sprite_available:
           available = False
       
       bg_color = "#f4a6c1" if available else "#cccccc"
       text_color = "white" if available else "#666666"
       
       btn = TouchButton(drink_frame, drink_name, 
                        lambda d=drink_id: select_drink(d) if available else None,
                        width=250, height=80, font_size=14, 
                        bg_color=bg_color, text_color=text_color)
       btn.pack()
       
       if not available:
           unavail_label = tk.Label(drink_frame, text="(Sprite not available)",
                                  font=("Helvetica", 10),
                                  bg="#f5f1ed", fg="red")
           unavail_label.pack(pady=2)
   
   # Configure grid weights
   drinks_frame.grid_columnconfigure(0, weight=1)
   drinks_frame.grid_columnconfigure(1, weight=1)
   
   # Back button
   back_frame = tk.Frame(main_frame, bg="#f5f1ed")
   back_frame.pack(fill="x", pady=20)
   
   TouchButton(back_frame, "← BACK", go_back, 
               width=150, height=60, font_size=16, bg_color="#ff7043").pack(anchor="w")
   
   root.after(100, lambda: root.focus_set())
   
   try:
       root.mainloop()
   except:
       pass
   
   if should_exit:
       return None
   
   return result[0]

def show_place_cup_screen(drink_name):
   """Show screen asking user to place cup"""
   global should_exit, waiting_for_cup
   
   def on_ready():
       nonlocal cup_placed
       cup_placed = True
       root.quit()
   
   def on_cancel():
       nonlocal cancelled
       cancelled = True
       root.quit()
   
   cup_placed = False
   cancelled = False
   waiting_for_cup = True
   
   root = get_root_window()
   clear_window(root)
   
   main_frame = tk.Frame(root, bg="#f5f1ed")
   main_frame.pack(fill="both", expand=True, padx=20, pady=20)
   
   # Instructions
   instruction_frame = tk.Frame(main_frame, bg="#f5f1ed")
   instruction_frame.pack(expand=True, fill="both")
   
   cup_icon = tk.Label(instruction_frame, text="🥤", 
                      font=("Helvetica", 72), bg="#f5f1ed")
   cup_icon.pack(pady=20)
   
   title_label = tk.Label(instruction_frame, text="Please place your cup",
                        font=("Helvetica", 28, "bold"),
                        bg="#f5f1ed", fg="#f4a6c1")
   title_label.pack(pady=(0, 10))
   
   drink_label = tk.Label(instruction_frame, text=f"Preparing: {drink_name}",
                        font=("Helvetica", 20),
                        bg="#f5f1ed", fg="#7dd3c0")
   drink_label.pack(pady=(0, 20))
   
   instruction_label = tk.Label(instruction_frame, 
                              text="Place your cup under the dispenser\nand press READY when done",
                              font=("Helvetica", 18),
                              bg="#f5f1ed", fg="#666666",
                              justify="center")
   instruction_label.pack(pady=20)
   
   # Status label for automatic detection
   status_label = tk.Label(instruction_frame, text="Waiting for cup...",
                         font=("Helvetica", 16),
                         bg="#f5f1ed", fg="#ffa726")
   status_label.pack(pady=10)
   
   # Buttons
   button_frame = tk.Frame(main_frame, bg="#f5f1ed")
   button_frame.pack(fill="x", pady=20)
   
   TouchButton(button_frame, "READY", on_ready, 
               width=200, height=70, font_size=20).pack(side="left", padx=20)
   
   TouchButton(button_frame, "CANCEL", on_cancel, 
               width=200, height=70, font_size=20, bg_color="#ff7043").pack(side="right", padx=20)
   
   # Auto-detect cup placement
   def check_cup_status():
       if not waiting_for_cup or should_exit:
           return
           
       dist = measure_distance()
       if dist <= TRIGGER_DISTANCE:
           status_label.config(text="Cup detected! 🎯", fg="#7dd3c0")
           # Auto-proceed after 2 seconds
           root.after(2000, lambda: on_ready() if waiting_for_cup else None)
       else:
           status_label.config(text="Waiting for cup...", fg="#ffa726")
           root.after(500, check_cup_status)
   
   root.after(100, lambda: root.focus_set())
   root.after(500, check_cup_status)
   
   try:
       root.mainloop()
   except:
       pass
   
   waiting_for_cup = False
   
   if should_exit:
       return None
   
   return "ready" if cup_placed else ("cancel" if cancelled else None)

def start_drink_preparation():
   """Start the drink preparation process"""
   global current_drink, mqtt_client
   
   if current_drink and mqtt_client:
       print(f"🍹 Starting preparation of {current_drink}")
       mqtt_client.publish(MQTT_TOPIC_DRINK_SELECTION, current_drink)
       show_making_drink_screen()

def show_making_drink_screen():
    """Show drink preparation screen with progress updates"""
    global should_exit, current_drink, mqtt_client
    
    root = get_root_window()
    clear_window(root)
    
    main_frame = tk.Frame(root, bg="#f5f1ed")
    main_frame.pack(fill="both", expand=True, padx=20, pady=20)
    
    # Animation frame
    animation_frame = tk.Frame(main_frame, bg="#f5f1ed")
    animation_frame.pack(expand=True, fill="both")
    
    # Animated icon
    icon_label = tk.Label(animation_frame, text="🔄", 
                         font=("Helvetica", 72), bg="#f5f1ed")
    icon_label.pack(pady=30)
    
    title_label = tk.Label(animation_frame, text="Making your drink...",
                          font=("Helvetica", 28, "bold"),
                          bg="#f5f1ed", fg="#f4a6c1")
    title_label.pack(pady=(0, 10))
    
    if current_drink and current_drink in DRINK_OPTIONS:
        drink_name = DRINK_OPTIONS[current_drink]
        drink_label = tk.Label(animation_frame, text=f"Preparing: {drink_name}",
                             font=("Helvetica", 20),
                             bg="#f5f1ed", fg="#7dd3c0")
        drink_label.pack(pady=(0, 20))
    
    # Add progress label that will be updated
    progress_label = tk.Label(animation_frame, text="Starting preparation...",
                            font=("Helvetica", 18),
                            bg="#f5f1ed", fg="#666666")
    progress_label.pack(pady=20)
    
    # Progress bar
    progress_bar = ttk.Progressbar(animation_frame, orient="horizontal", 
                                 length=400, mode="determinate")
    progress_bar.pack(pady=10)
    
    # Status update function
    def update_progress(status, value=None):
        try:
            progress_label.config(text=status)
            if value is not None:
                progress_bar['value'] = value
            root.update()
        except:
            pass
    
    # Animate the icon
    def animate_icon():
        if should_exit:
            return
        try:
            icons = ["🔄", "🌀", "⚡", "💫"]
            current_icon = icons[int(time.time()) % len(icons)]
            icon_label.config(text=current_icon)
            root.after(500, animate_icon)
        except tk.TclError:
            return
    
    # MQTT message handler for this screen
    def handle_drink_progress(client, userdata, msg):
        message = msg.payload.decode()
        print(f"Drink progress: {message}")
        
        if "Dispensing" in message:
            update_progress(message, 25)
        elif "Mixing" in message:
            update_progress(message, 50)
        elif "Finalizing" in message:
            update_progress(message, 75)
        elif "complete" in message.lower() and "stir" not in message.lower():
            update_progress("Drink preparation complete!", 100)
    
    # Subscribe to progress updates
    mqtt_client.message_callback_add(MQTT_TOPIC_MOTOR_STATUS, handle_drink_progress)
    
    root.after(100, lambda: root.focus_set())
    animate_icon()
    
    # Keep the screen open until drink is complete
    while not should_exit and not system_ready:
        root.update()
        time.sleep(0.1)
    
    # Unsubscribe the progress handler
    mqtt_client.message_callback_remove(MQTT_TOPIC_MOTOR_STATUS)

def show_stir_question_mqtt():
    """Show stir question when drink is complete"""
    global should_exit, waiting_for_stir_response, mqtt_client, system_ready, recognized_user, current_drink
    
    def send_response(response):
        """Helper function to send MQTT response with error handling"""
        try:
            if not mqtt_client.is_connected():
                print("MQTT client disconnected, reconnecting...")
                mqtt_client.reconnect()
                
            print(f"[DEBUG] Sending stir response: {response}")
            info = mqtt_client.publish(MQTT_TOPIC_STIR_RESPONSE, response, qos=1)
            
            # Wait for message to be published (max 2 seconds)
            start_time = time.time()
            while not info.is_published() and (time.time() - start_time) < 2:
                time.sleep(0.1)
                
            if info.is_published():
                print(f"[DEBUG] Response '{response}' successfully published")
                return True
            else:
                print("[ERROR] Failed to publish response")
                return False
                
        except Exception as e:
            print(f"[ERROR] Error sending response: {str(e)}")
            return False

    def on_yes():
        if send_response("yes"):
            # Print receipt before showing stirring screen
            if recognized_user and recognized_user[0]:
                name = recognized_user[0]
            else:
                print_receipt(name or "Customer", DRINK_OPTIONS.get(current_drink, "Unknown Drink"))
            
            show_stirring_screen()
            # Add visual feedback while waiting
            status_label.config(text="Stirring in progress...", fg="blue")
            root.update()
            # Wait for completion
            start_time = time.time()
            while (not should_exit and not system_ready and 
                  (time.time() - start_time) < 30):  # 30 second timeout
                root.update()
                time.sleep(0.1)
                
            if not system_ready:
                status_label.config(text="Stirring timed out!", fg="red")
                root.update()
                time.sleep(2)
        root.quit()

    def on_no():
        if send_response("no"):
            # Print receipt before showing final screen
            if recognized_user and recognized_user[0]:
                name = recognized_user[0]
            else:
                print_receipt(name or "Customer", DRINK_OPTIONS.get(current_drink, "Unknown Drink"))
            
            show_drink_ready_final()
        root.quit()

    root = get_root_window()
    clear_window(root)
    
    main_frame = tk.Frame(root, bg="#f5f1ed")
    main_frame.pack(fill="both", expand=True, padx=20, pady=20)
    
    # Header
    header_frame = tk.Frame(main_frame, bg="#f5f1ed")
    header_frame.pack(fill="x", pady=(20, 30))
    
    icon_label = tk.Label(header_frame, text="🥄", 
                         font=("Helvetica", 72), bg="#f5f1ed")
    icon_label.pack()
    
    title_label = tk.Label(header_frame, text="Drink Complete!",
                         font=("Helvetica", 28, "bold"),
                         bg="#f5f1ed", fg="#7dd3c0")
    title_label.pack(pady=(10, 5))
    
    question_label = tk.Label(header_frame, text="Would you like me to stir your drink?",
                            font=("Helvetica", 22),
                            bg="#f5f1ed", fg="#333333")
    question_label.pack(pady=(0, 30))
    
    
    # Status label for feedback
    status_label = tk.Label(main_frame, text="", 
                          font=("Helvetica", 16),
                          bg="#f5f1ed", fg="#666666")
    status_label.pack(pady=10)
    
    # Buttons
    button_frame = tk.Frame(main_frame, bg="#f5f1ed")
    button_frame.pack(expand=True)
    
    TouchButton(button_frame, "YES, STIR IT", on_yes, 
                width=250, height=90, font_size=18).pack(pady=15)
    
    TouchButton(button_frame, "NO, THANKS", on_no, 
                width=250, height=90, font_size=18, bg_color="#ff7043").pack(pady=15)
    
    root.after(100, lambda: root.focus_set())
    
    try:
        root.mainloop()
    except Exception as e:
        print(f"UI Error: {e}")
    
    waiting_for_stir_response = False
    system_ready = True
    
def show_stirring_screen():
   """Show stirring animation"""
   global should_exit
   
   root = get_root_window()
   clear_window(root)
   
   main_frame = tk.Frame(root, bg="#f5f1ed")
   main_frame.pack(fill="both", expand=True, padx=20, pady=20)
   
   animation_frame = tk.Frame(main_frame, bg="#f5f1ed")
   animation_frame.pack(expand=True, fill="both")
   
   icon_label = tk.Label(animation_frame, text="🌪️", 
                        font=("Helvetica", 72), bg="#f5f1ed")
   icon_label.pack(pady=30)
   
   title_label = tk.Label(animation_frame, text="Stirring your drink...",
                        font=("Helvetica", 28, "bold"),
                        bg="#f5f1ed", fg="#f4a6c1")
   title_label.pack(pady=(0, 20))
   
   status_label = tk.Label(animation_frame, text="Please wait while I mix it perfectly",
                         font=("Helvetica", 18),
                         bg="#f5f1ed", fg="#666666")
   status_label.pack(pady=20)

   
   # Animate stirring
   def animate_stir():
        if should_exit:
            return
        try:
            icons = ["🌪️", "🌀", "💫", "⭐"]
            current_icon = icons[int(time.time() * 2) % len(icons)]
            icon_label.config(text=current_icon)
            root.after(300, animate_stir)
        except tk.TclError:
            # Widget no longer exists, stop animation
            return
   
   root.after(100, lambda: root.focus_set())
   animate_stir()
   
   root.update()

def show_drink_ready_screen():
   """Show initial drink ready screen (before stir question)"""
   global should_exit
   
   root = get_root_window()
   clear_window(root)
   
   main_frame = tk.Frame(root, bg="#f5f1ed")
   main_frame.pack(fill="both", expand=True, padx=20, pady=20)
   
   ready_frame = tk.Frame(main_frame, bg="#f5f1ed")
   ready_frame.pack(expand=True, fill="both")
   
   icon_label = tk.Label(ready_frame, text="✅", 
                        font=("Helvetica", 72), bg="#f5f1ed")
   icon_label.pack(pady=30)
   
   title_label = tk.Label(ready_frame, text="Drink Complete!",
                        font=("Helvetica", 28, "bold"),
                        bg="#f5f1ed", fg="#7dd3c0")
   title_label.pack(pady=(0, 20))
   
   message_label = tk.Label(ready_frame, text="Your delicious drink is ready!\nWaiting for stir option...",
                          font=("Helvetica", 18),
                          bg="#f5f1ed", fg="#666666",
                          justify="center")
   message_label.pack(pady=20)
   
   root.after(100, lambda: root.focus_set())
   root.update()

def show_drink_ready_final():
   """Show final drink ready screen"""
   global should_exit, current_drink, system_ready
   
   def on_continue():
       result[0] = True
       root.quit()
   
   result = [None]
   
   root = get_root_window()
   clear_window(root)
   
   main_frame = tk.Frame(root, bg="#f5f1ed")
   main_frame.pack(fill="both", expand=True, padx=20, pady=20)
   
   # Success message
   success_frame = tk.Frame(main_frame, bg="#f5f1ed")
   success_frame.pack(expand=True, fill="both")
   
   icon_label = tk.Label(success_frame, text="🎉", 
                        font=("Helvetica", 72), bg="#f5f1ed")
   icon_label.pack(pady=20)
   
   title_label = tk.Label(success_frame, text="Enjoy your drink!",
                        font=("Helvetica", 32, "bold"),
                        bg="#f5f1ed", fg="#7dd3c0")
   title_label.pack(pady=(0, 10))
   
   subtitle_label = tk.Label(success_frame, text="Your perfect beverage is ready to enjoy",
                           font=("Helvetica", 18),
                           bg="#f5f1ed", fg="#666666")
   subtitle_label.pack(pady=(0, 30))
   
   thanks_label = tk.Label(success_frame, text="Thank you for using BARBOT!",
                         font=("Helvetica", 20),
                         bg="#f5f1ed", fg="#f4a6c1")
   thanks_label.pack(pady=20)
   
   # Continue button
   button_frame = tk.Frame(main_frame, bg="#f5f1ed")
   button_frame.pack(fill="x", pady=20)
   
   TouchButton(button_frame, "CONTINUE", on_continue, 
               width=250, height=70, font_size=20).pack(anchor="center")
   
   root.after(100, lambda: root.focus_set())
   
   try:
       root.mainloop()
   except:
       pass
   
   # Reset system state
   current_drink = None
   system_ready = True
   
   if should_exit:
       return

def show_processing_screen(title, message, duration):
   """Show processing screen with animated loading"""
   global should_exit
   
   root = get_root_window()
   clear_window(root)
   
   main_frame = tk.Frame(root, bg="#f5f1ed")
   main_frame.pack(fill="both", expand=True)
   
   title_label = tk.Label(main_frame, text=title,
                          font=("Helvetica", 24, "bold"),
                          bg="#f5f1ed", fg="#f4a6c1")
   title_label.pack(expand=True)
   
   message_label = tk.Label(main_frame, text=message,
                           font=("Helvetica", 18),
                           bg="#f5f1ed", fg="#666666")
   message_label.pack(pady=(0, 40))
   
   root.update()
   time.sleep(duration)

def show_error_screen(message):
   """Show error message with retry option"""
   global should_exit
   
   def on_ok():
       root.quit()
   
   root = get_root_window()
   clear_window(root)
   
   main_frame = tk.Frame(root, bg="#f5f1ed")
   main_frame.pack(fill="both", expand=True, padx=20, pady=20)
   
   error_frame = tk.Frame(main_frame, bg="#f5f1ed")
   error_frame.pack(expand=True, fill="both")
   
   icon_label = tk.Label(error_frame, text="⚠️", 
                        font=("Helvetica", 72), bg="#f5f1ed")
   icon_label.pack(pady=20)
   
   title_label = tk.Label(error_frame, text="Oops!",
                        font=("Helvetica", 28, "bold"),
                        bg="#f5f1ed", fg="#ff6b6b")
   title_label.pack(pady=(20, 10))
   
   message_label = tk.Label(error_frame, text=message,
                          font=("Helvetica", 18),
                          bg="#f5f1ed", fg="#666666",
                          wraplength=600, justify="center")
   message_label.pack(pady=(0, 30))
   
   TouchButton(error_frame, "OK", on_ok, 
               width=200, height=70, font_size=20).pack()
   
   root.after(100, lambda: root.focus_set())
   
   try:
       root.mainloop()
   except:
       pass

def show_goodbye_screen():
   """Show goodbye message"""
   global should_exit
   
   root = get_root_window()
   clear_window(root)
   
   main_frame = tk.Frame(root, bg="#f5f1ed")
   main_frame.pack(fill="both", expand=True)
   
   goodbye_frame = tk.Frame(main_frame, bg="#f5f1ed")
   goodbye_frame.pack(expand=True, fill="both")
   
   icon_label = tk.Label(goodbye_frame, text="👋", 
                        font=("Helvetica", 72), bg="#f5f1ed")
   icon_label.pack(pady=30)
   
   title_label = tk.Label(goodbye_frame, text="Thank you!",
                        font=("Helvetica", 32, "bold"),
                        bg="#f5f1ed", fg="#f4a6c1")
   title_label.pack(expand=True)
   
   subtitle_label = tk.Label(goodbye_frame, text="Come back soon for more delicious drinks!",
                           font=("Helvetica", 20),
                           bg="#f5f1ed", fg="#666666")
   subtitle_label.pack(pady=(0, 40))
   
   root.update()
   time.sleep(2)

# === MAIN APPLICATION LOGIC ===
def main():
    """Main BARBOT application loop"""
    global should_exit, root_window, current_drink, recognized_user, system_ready

    print("🤖 Starting BARBOT...")
    def play_background_music():
        while not should_exit:
            try:
                subprocess.run([
                    "mpg123", 
                    "--loop", "-1",  # Infinite loop
                    "/home/barbot/BARBOT_Final/music/system_theme.mp3"
                ], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error playing background music: {e}")
            except Exception as e:
                print(f"Unexpected error in music playback: {e}")
    def stop_background_music():
        global should_exit, music_process
        should_exit = True
        if music_process:
            try:
                # Send SIGTERM to the music process
                music_process.terminate()
                # Wait a bit for it to terminate
                music_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                # If it didn't terminate, force kill it
                music_process.kill()
            except Exception as e:
                print(f"Error stopping music: {e}")




    # Initialize all systems
    init_all_systems()
   
    music_thread = threading.Thread(target=play_background_music, daemon=True)
    music_thread.start()
    try:
        # Initialize the main window
        root_window = get_root_window()

        while not should_exit:
            try:
                # Reset state
                current_drink = None
                recognized_user = None
                system_ready = True

                # Show main welcome screen
                choice = show_welcome_screen()

                if should_exit:
                    break

                if choice == "face_recognition":
                    # Face recognition flow
                    print("👤 Starting face recognition...")
                    name, drink = capture_and_recognize_face()

                    if should_exit:
                        break

                    if name and drink:
                        # User recognized
                        print(f"✅ User recognized: {name}")
                        user_choice = show_recognized_user_screen(name, drink)

                        if should_exit:
                            break

                        if user_choice == "favorite":
                            # Order their favorite drink
                            current_drink = None
                            # Find drink ID by name
                            for drink_id, drink_name in DRINK_OPTIONS.items():
                                if drink.lower() in drink_name.lower():
                                    current_drink = drink_id
                                    break
                            
                            if current_drink:
                                drink_name = DRINK_OPTIONS[current_drink]
                                cup_response = show_place_cup_screen(drink_name)
                                
                                if should_exit:
                                    break
                                
                                if cup_response == "ready":
                                    start_drink_preparation()
                                    # Wait for drink completion (handled by MQTT)
                                    while not should_exit and not system_ready:
                                        time.sleep(0.5)
                                        root_window.update()
                                    
                            else:
                                show_error_screen("Sorry, your favorite drink is not available.")
                        
                        elif user_choice == "new_drink":
                            # Show menu for recognized user
                            selected_drink = show_drink_menu()
                            
                            if should_exit:
                                break
                            
                            if selected_drink and selected_drink != "back":
                                current_drink = selected_drink
                                drink_name = DRINK_OPTIONS[selected_drink]
                                cup_response = show_place_cup_screen(drink_name)
                                
                                if should_exit:
                                    break
                                
                                if cup_response == "ready":
                                    start_drink_preparation()
                                    # Wait for drink completion
                                    while not should_exit and not system_ready:
                                        time.sleep(0.5)
                                        root_window.update()
                                    # print_receipt(name, drink_name)
                    else:
                        # User not recognized - offer registration
                        show_error_screen("I don't recognize you yet.\nWould you like to register as a new user?")
                        
                        if should_exit:
                            break
                        
                        # Go to registration flow
                        name, drink = show_name_input_screen()
                        
                        if should_exit:
                            break
                        
                        if name and drink:
                            success = register_new_user(name, drink)
                            
                            if should_exit:
                                break
                            
                            if success:
                                # Offer to order a drink now
                                selected_drink = show_drink_menu()
                                
                                if should_exit:
                                    break
                                
                                if selected_drink and selected_drink != "back":
                                    current_drink = selected_drink
                                    drink_name = DRINK_OPTIONS[selected_drink]
                                    cup_response = show_place_cup_screen(drink_name)
                                    
                                    if should_exit:
                                        break
                                    
                                    if cup_response == "ready":
                                        start_drink_preparation()
                                        # Wait for drink completion
                                        while not should_exit and not system_ready:
                                            time.sleep(0.5)
                                            root_window.update()
                                        #print_receipt(name, drink_name)

                elif choice == "manual_order":
                    # Manual drink ordering
                    print("🍹 Starting manual order...")
                    selected_drink = show_drink_menu()
                    
                    if should_exit:
                        break
                    
                    if selected_drink and selected_drink != "back":
                        current_drink = selected_drink
                        drink_name = DRINK_OPTIONS[selected_drink]
                        cup_response = show_place_cup_screen(drink_name)
                        
                        if should_exit:
                            break
                        
                        if cup_response == "ready":
                            start_drink_preparation()
                            # Wait for drink completion
                            while not should_exit and not system_ready:
                                time.sleep(0.5)
                                root_window.update()
                            #print_receipt("Customer", drink_name)

                elif choice == "register":
                    # New user registration
                    print("📝 Starting new user registration...")
                    name, drink = show_name_input_screen()
                    
                    if should_exit:
                        break
                    
                    if name and drink:
                        success = register_new_user(name, drink)
                        
                        if should_exit:
                            break
                        
                        if success:
                            # Offer to order a drink now
                            selected_drink = show_drink_menu()
                            
                            if should_exit:
                                break
                            
                            if selected_drink and selected_drink != "back":
                                current_drink = selected_drink
                                drink_name = DRINK_OPTIONS[selected_drink]
                                cup_response = show_place_cup_screen(drink_name)
                                
                                if should_exit:
                                    break
                                
                                if cup_response == "ready":
                                    start_drink_preparation()
                                    # Wait for drink completion
                                    while not should_exit and not system_ready:
                                        time.sleep(0.5)
                                        root_window.update()
                                    #print_receipt(name, drink_name)

                elif choice is None or choice == "exit":
                    # Exit application
                    should_exit = True
                    break
                
                # Handle stir response if needed
                if waiting_for_stir_response and not should_exit:
                    show_stir_question_mqtt()
                
                # Small delay before next iteration
                if not should_exit:
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                show_error_screen(f"System error: {str(e)}")
                time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n🛑 Shutting down BARBOT via Ctrl+C...")
        should_exit = True
        stop_background_music()
        music_thread.join(timeout=1)  # Wait for thread to finish
    
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        show_error_screen(f"Fatal system error: {str(e)}")
        stop_background_music()
        music_thread.join(timeout=1)  # Wait for thread to finish
    
    finally:
        # Cleanup
        print("🧹 Cleaning up systems...")
        
        try:
            if picam:
                picam.stop()
                picam.close()
                print("✅ Camera cleaned up")
        except:
            print("❌ Error cleaning up camera")
        
        try:
            if mqtt_client:
                mqtt_client.loop_stop()
                mqtt_client.disconnect()
                print("✅ MQTT disconnected")
        except:
            print("❌ Error disconnecting MQTT")
        
        try:
            GPIO.cleanup()
            print("✅ GPIO cleaned up")
        except:
            print("❌ Error cleaning up GPIO")
        
        try:
            pygame.quit()
            print("✅ Pygame cleaned up")
        except:
            print("❌ Error cleaning up pygame")
        
        try:
            if root_window and root_window.winfo_exists():
                root_window.destroy()
                print("✅ GUI cleaned up")
        except:
            print("❌ Error cleaning up GUI")
        
        show_goodbye_screen()
        print("🤖 BARBOT shutdown complete")

if __name__ == "__main__":
   main()