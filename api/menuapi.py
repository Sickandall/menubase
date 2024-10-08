from flask import Flask, request,Response, jsonify, render_template
from flask import Flask, send_file, jsonify, stream_with_context
import threading
from twilio.rest import Client 
from instagrapi import Client as InstaClient
from twilio.rest import Client as TwilioClient
import smtplib
import speech_recognition as sr
import http.client
import json
import pickle
import subprocess
import pyttsx3
import getpass
import schedule
import boto3
from instabot import Bot
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from threading import Thread
from googlesearch import search
import cv2
import os
from ai21 import AI21Client
from ai21.models.chat import ChatMessage
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import binascii
app = Flask(__name__)

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/profile.html')
def about():
    return render_template('profile.html')

@app.route('/project.html')
def project():
    return render_template('project.html')

@app.route('/skills.html')
def skills():
    return render_template('skills.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

account_sid = 'AC96a802982efe963b730e44fc203e6f8d'
auth_token = 'b36843b188808dcd3307190dc9a00503'
client = Client(account_sid, auth_token)

@app.route('/text_message', methods=['POST'])
def send_text_sms():
    try:
        # Get the message from the frontend
        message_body = request.json.get('message')
        recipient = request.json.get('to')

        # Send the SMS using Twilio
        message = client.messages.create(
            to=recipient,
            from_="+12202355382",
            body=message_body
        )

        return jsonify({'status': 'success', 'message_sid': message.sid})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500   
account_sid = 'AC96a802982efe963b730e44fc203e6f8d'
auth_token = 'b36843b188808dcd3307190dc9a00503'
@app.route("/make_phone_call", methods=['POST'])
def make_phone_call():
    try:
        data = request.get_json()
        to_number = data.get('to')
        twilio_number = '+12202355382'
        twiml_url = 'http://demo.twilio.com/docs/voice.xml'
        
        call = client.calls.create(
            to=to_number,
            from_=twilio_number,
            url=twiml_url
        )
        
        return jsonify({"status": "success", "call_sid": call.sid}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500  
@app.route('/post_photo', methods=['POST'])
def instagram_post():
    username = request.form.get('username')
    password = request.form.get('password')
    caption = request.form.get('caption')
    photo = request.files.get('photo')

    try:
        # Save the photo temporarily
        photo_path = f"./{photo.filename}"
        photo.save(photo_path)

        # Login to Instagram and post the photo
        insta_client = InstaClient()
        insta_client.login(username, password)
        post = insta_client.photo_upload(photo_path, caption)

        return jsonify({'status': 'success', 'post_url': post.dict()['url']})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})
@app.route('/send_whatsapp', methods=['POST'])
def send_whatsapp_message():
    try:
        data = request.json
        recipient = data['messages'][0]['to']
        template_name = data['messages'][0]['content']['templateName']
        placeholders = data['messages'][0]['content']['templateData']['body']['placeholders']

        conn = http.client.HTTPSConnection("n8yxz2.api.infobip.com")
        payload = json.dumps({
            "messages": [
                {
                    "from": "447860099299",  # Your Infobip phone number
                    "to": recipient,  # Recipient's phone number
                    "messageId": "unique-message-id",  # Generate a unique ID for the message
                    "content": {
                        "templateName": template_name,
                        "templateData": {
                            "body": {
                                "placeholders": placeholders
                            }
                        },
                        "language": "en"
                    }
                }
            ]
        })

        headers = {
            'Authorization': 'App d99e580de86f448d79f2922e2c116e0c-1c0aed72-1130-4918-b310-3cb20638085c',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        conn.request("POST", "/whatsapp/1/message/template", payload, headers)
        res = conn.getresponse()
        data = res.read()

        # Decode the response and check for success
        response_data = json.loads(data.decode("utf-8"))
        if res.status == 200:
            return jsonify({"status": "success", "response": response_data}), 200
        else:
            return jsonify({"status": "error", "message": response_data}), 500

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/send_email', methods=['POST'])
def send_email():
    try:
        # Get the form data from the frontend
        email = "yugpratap2003@gmail.com"
        password = request.json.get('password')
        receiver_email = request.json.get('receiver_email')
        subject = request.json.get('subject')
        message = request.json.get('message')
        
        # Send the email
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(email, password)
        server.sendmail(email, receiver_email, f"Subject: {subject}\n\n{message}")
        return jsonify({'status': 'success', 'message': f"Email has been sent to {receiver_email}"})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Email configuration
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SMTP_USERNAME = 'your_email@gmail.com'
SMTP_PASSWORD = 'your_password'

def schedul_mail(receiver_email, subject, body):
    sender_email = SMTP_USERNAME

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        print(f'Email sent successfully to {receiver_email}')
    except Exception as e:
        print(f'Failed to send email. Error: {str(e)}')
    finally:
        server.quit()

# Scheduler function to run continuously in the background
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(60)  # Wait for 1 minute

# Start the scheduler in a separate thread
scheduler_thread = Thread(target=run_scheduler)
scheduler_thread.start()

@app.route('/schedule_email', methods=['POST'])
def schedule_email():
    data = request.get_json()
    schedule_time = data['scheduleTime']
    subject = data['subject']
    message = data['message']
    receiver_email = data['receiverEmail']

    # Schedule the email to be sent at the specified time
    schedule.every().day.at(schedule_time).do(schedul_mail, receiver_email=receiver_email, subject=subject, body=message)

    return jsonify({'status': 'success', 'message': 'Email scheduled successfully'})

@app.route('/google_search', methods=['POST'])
def google_search():
    try:
        data = request.get_json()
        query = data.get('query')
        num_results = int(data.get('num_results', 5))  # Ensure num_results is an integer
        
        if not query:
            return jsonify({'status': 'error', 'message': 'Query is required'}), 400
        
        # Convert generator to list
        search_results = list(search(query, num_results=num_results))
        
        return jsonify({'status': 'success', 'results': search_results})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
@app.route('/launch_ec2', methods=['POST'])
def launch_ec2():
    try:
        session = boto3.Session(
            aws_access_key_id='AKIA3FLD36SYJOEK6G5A',
            aws_secret_access_key='X5ngGTFwNeNYzINP9ueFujyYs/qhZyRWDVaO0oRl',
            region_name='ap-south-1'  # replace with your region
        )

        ec2 = session.resource('ec2')
        instances = ec2.create_instances(
            InstanceType='t2.micro',
            ImageId='ami-0ec0e125bb6c6e8ec',
            MaxCount=1,
            MinCount=1
        )

        return jsonify({"status": "success", "message": "EC2 Instance launched"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
@app.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    try:
        # Get the text from the request
        text = request.json.get('text')
        
        # Initialize the text-to-speech engine
        engine = pyttsx3.init()
        
        # Speak the text
        engine.say(text)
        engine.runAndWait()
        
        return jsonify({"status": "success", "message": "Text spoken successfully"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
def apply_filter(img, filter_type):
    if filter_type == 0:
        return img  # No filter
    elif filter_type == 1:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Grayscale filter
    elif filter_type == 2:
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        return cv2.transform(img, kernel)  # Sepia filter
    elif filter_type == 3:
        return cv2.bitwise_not(img)  # Negative filter
    elif filter_type == 4:
        return cv2.GaussianBlur(img, (15, 15), 0)  # Gaussian blur filter
    elif filter_type == 5:
        return cv2.Canny(img, 100, 200)  # Canny edge detection
    else:
        return img  # Default to no filter

@app.route('/start_filter', methods=['POST'])
def start_filter():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(staticMode=False, maxHands=1, detectionCon=0.8, minTrackCon=0.5)
    prev_fingers = -1

    while True:
        success, img = cap.read()
        if not success:
            break

        hands, img = detector.findHands(img, draw=True)
        if hands:
            hand = hands[0]
            fingers = detector.fingersUp(hand).count(1)
            if fingers != prev_fingers:
                prev_fingers = fingers
                print(f'Number of fingers: {fingers}')

        filtered_img = apply_filter(img, prev_fingers)
        cv2.imshow("Image", filtered_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return jsonify({'status': 'success'})

def capture_photo(filename='captured_photo.jpg'):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return False, "Could not open webcam."
    
    ret, frame = cap.read()
    cap.release()

    if ret:
        cv2.imwrite(filename, frame)
        return True, f"Photo captured and saved as {filename}"
    else:
        return False, "Failed to capture photo."

def send_email_with_photo(receiver_email, sender_email, sender_password, filename='captured_photo.jpg'):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = 'Photo captured'

    with open(filename, 'rb') as fp:
        img = MIMEImage(fp.read())
    img.add_header('Content-Disposition', 'attachment', filename=filename)
    msg.attach(img)

    body = MIMEText('Please find attached the photo that was captured.')
    msg.attach(body)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)  # Using Gmail's SMTP server
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        return True, "Email sent successfully"
    except Exception as e:
        return False, f"Failed to send email. Error: {str(e)}"

@app.route('/capture_and_send_email', methods=['POST'])
def capture_and_send_email():
    data = request.json
    sender_email = data['sender_email']
    sender_password = data['sender_password']
    receiver_email = data['receiver_email']

    success, message = capture_photo()
    if not success:
        return jsonify(status='failed', message=message)

    success, message = send_email_with_photo(receiver_email, sender_email, sender_password)
    if not success:
        return jsonify(status='failed', message=message)

    return jsonify(status='success', message=message)
# Function to capture a photo using the webcam
def capture_photo(filename='captured_photo.jpg'):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return False
    
    ret, frame = cap.read()
    cap.release()

    if ret:
        cv2.imwrite(filename, frame)
        print(f"Photo captured and saved as {filename}")
        return filename
    else:
        print("Error: Failed to capture photo.")
        return False

# Function to upload the photo to Instagram
def send_photo_to_instagram(username, password, filename='captured_photo.jpg'):
    bot = Bot()
    bot.login(username=username, password=password)
    bot.upload_photo(filename)
@app.route('/upload_instagram_photo', methods=['POST'])
def upload_instagram_photo():
    data = request.get_json()
    username = data['username']
    password = data['password']
    
    filename = capture_photo()
    if filename:
        send_photo_to_instagram(username, password, filename)
        return jsonify({"message": "Photo uploaded to Instagram successfully!"})
    else:
        return jsonify({"message": "Failed to capture photo."}), 500
@app.route('/apply_image_filters', methods=['POST'])
def apply_image_filters():
    if 'image' not in request.files:
        return jsonify({"message": "No image file uploaded."}), 400
    
    image_file = request.files['image']
    image_path = os.path.join('static/uploads', image_file.filename)
    image_file.save(image_path)

    # Load the image
    image = cv2.imread(image_path)

    # Convert color image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join('static/output', 'grayscale_image.jpg'), grayscale_image)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
    cv2.imwrite(os.path.join('static/output', 'blurred_image.jpg'), blurred_image)

    # Apply edge detection
    edges = cv2.Canny(image, 100, 200)
    cv2.imwrite(os.path.join('static/output', 'edges_image.jpg'), edges)

    # Apply sharpening filter
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    cv2.imwrite(os.path.join('static/output', 'sharpened_image.jpg'), sharpened_image)

    return jsonify({"message": "Image filters applied successfully. Check output folder for results."})

@app.route('/launch_webserver', methods=['POST'])
def launch_webserver():
    try:
        # Build the Docker image
        subprocess.run(["docker", "build", "-t", "centos-webserver", "."], check=True)
        
        # Run the Docker container
        subprocess.run(["docker", "run", "-d", "-p", "8080:80", "centos-webserver"], check=True)
        
        return jsonify({"message": "Web server launched successfully on CentOS 7!"})
    except subprocess.CalledProcessError as e:
        return jsonify({"message": f"Failed to launch web server: {str(e)}"}), 500
# Function for live video capture and face crop
def live_video_capture_and_crop():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cropped_face = frame[y:y+h, x:x+w]
            cv2.imshow("Cropped Face", cropped_face)

        cv2.imshow("Live Video (Detected Faces)", frame)

        if cv2.waitKey(1) == 13:
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/live_video_capture_and_crop')
def live_video_feed():
    live_video_capture_and_crop()
    return "Live Video Capture and Face Crop Completed"

# Function to capture a photo using the webcam
def capture_photo(filename='captured_photo.jpg'):
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, filename)
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return False

    ret, frame = cap.read()
    cap.release()

    if ret:
        cv2.imwrite(filepath, frame)
        print(f"Photo captured and saved as {filepath}")
        return True
    else:
        print("Error: Failed to capture photo.")
        return False

@app.route('/capture_photo', methods=['GET'])
def capture_photo_endpoint():
    filename = 'captured_photo.jpg'
    if capture_photo(filename):
        return jsonify({"message": "Photo captured successfully and saved in output folder."}), 200
    else:
        return jsonify({"error": "Failed to capture photo"}), 500
# Function to speak text
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def listen_and_recognize():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        yield "Listening...\n"
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)

    try:
        yield "Recognizing...\n"
        text = recognizer.recognize_google(audio)
        yield f"You said: {text}\n"
        return text
    except sr.UnknownValueError:
        yield "Sorry, I did not understand that.\n"
        return ""
    except sr.RequestError as e:
        yield f"Request error: {e}\n"
        return ""

@app.route('/speech_to_text', methods=['GET'])
def speech_to_text():
    @stream_with_context
    def generate():
        recognized_text = ""
        for message in listen_and_recognize():
            yield f"data:{message}\n\n"
            if "You said:" in message:
                recognized_text = message.split(":")[1].strip()
        
        if recognized_text:
            speak_text(recognized_text)

    return Response(generate(), mimetype='text/event-stream')

# Function to run a train in the terminal
def run_train_in_terminal():
    try:
        def run_in_new_terminal(command, terminal='wsl'):
            try:
                if terminal == 'wsl':
                    # Use WSL to run the command
                    subprocess.run(['wsl', command])
                else:
                    print(f"Unsupported terminal: {terminal}")
            except FileNotFoundError:
                print(f"{terminal} is not installed. Please install it first.")
            except subprocess.CalledProcessError as e:
                print(f"Command failed with return code: {e.returncode}")

        # Run the train animation command
        run_in_new_terminal('sl')
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

@app.route('/run_train', methods=['GET'])
def run_train():
    if run_train_in_terminal():
        return jsonify({"message": "Train is running in the terminal."})
    else:
        return jsonify({"error": "Failed to run the train."}), 500
# Route to start VLC streaming
@app.route('/start_vlc_stream', methods=['POST'])
def start_vlc_stream():
    data = request.json
    video_path = data.get('video_path')
    stream_url = data.get('stream_url')

    if not video_path or not stream_url:
        return jsonify({"error": "Video path and stream URL are required."}), 400

    # Command to launch VLC with streaming options
    command = f'cvlc {video_path} --sout "#transcode{{vcodec=h264,acodec=mp3}}:rtp{{dst={stream_url},port=5004,mux=ts}}"'
    
    try:
        # Run the command in a new terminal window
        subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', command])
        return jsonify({"message": "VLC streaming started successfully."})
    except Exception as e:
        return jsonify({"error": f"Failed to start VLC stream: {str(e)}"}), 500

# Route for photo capture and transfer
@app.route('/capture_and_transfer', methods=['POST'])
def capture_and_transfer():
    try:
        # Capture photo using a Linux command
        photo_path = os.path.join('output', 'captured_photo.jpg')
        subprocess.run(['fswebcam', '-r', '1280x720', '--jpeg', '85', '-D', '1', photo_path])

        # Transfer the photo to a remote server
        remote_user = request.json['remote_user']
        remote_host = request.json['remote_host']
        remote_path = request.json['remote_path']

        transfer_command = f"scp {photo_path} {remote_user}@{remote_host}:{remote_path}"
        subprocess.run(transfer_command, shell=True)

        return jsonify({"status": "success", "message": "Photo captured and transferred successfully!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
# live sreaming in linux
@app.route('/start_live_streaming', methods=['POST'])
def start_live_streaming():
    data = request.get_json()
    stream_url = data['stream_url']
    camera_device = data['camera_device']

    try:
        # Command to start live streaming using FFmpeg
        command = f"ffmpeg -f v4l2 -i {camera_device} -f mpegts {stream_url}"
        subprocess.Popen(command, shell=True)
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
# find command shortcut key
# Dictionary of common Linux shortcut commands
shortcuts = {
    'copy': 'Ctrl + C',
    'paste': 'Ctrl + V',
    'cut': 'Ctrl + X',
    'undo': 'Ctrl + Z',
    'redo': 'Ctrl + Shift + Z',
    'find': 'Ctrl + F',
    'save': 'Ctrl + S',
    'open terminal': 'Ctrl + Alt + T',
    'close terminal': 'Ctrl + D',
    'new tab': 'Ctrl + Shift + T',
    'switch tab': 'Ctrl + PgUp/PgDn',
    'quit': 'Ctrl + Q',
    'log out': 'Ctrl + Alt + Del',
    'lock screen': 'Ctrl + Alt + L',
    'screenshot': 'PrtScn',
    'fullscreen': 'F11'
}

@app.route('/find_shortcut', methods=['POST'])
def find_shortcut():
    data = request.json
    keyword = data['keyword'].lower()  # Convert keyword to lowercase

    # Find the shortcut in the dictionary
    shortcut = shortcuts.get(keyword)
    
    if shortcut:
        return jsonify({"command": keyword, "shortcut": shortcut})
    else:
        return jsonify({"error": "Command not found"}), 404
    
    
# Modify the Speed and Voice of the espeak Command in Linux  
@app.route('/modify_espeak', methods=['POST'])
def modify_espeak():
    data = request.json
    text = data.get('text')
    speed = data.get('speed', '150')
    voice = data.get('voice', 'default')

    # Use WSL to execute espeak-ng command
    command = f'wsl espeak-ng -v {voice} -s {speed} "{text}"'
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode == 0:
        return jsonify({"status": "success"})
    else:
        return jsonify({"status": "error", "error": result.stderr})
        
#chating with ping command
@app.route('/ping_chating', methods=['POST'])
def chatPing():
    def hex_to_text(hex_str):
        try:
            bytes_object = bytes.fromhex(hex_str)
            return bytes_object.decode("utf-8")
        except ValueError:
            return "Invalid hex value received."

    def text_to_hex(text):
        hex_str = binascii.hexlify(text.encode()).decode()
        return hex_str
    
    def send_message(ip, message):
        hex_message = text_to_hex(message)
        command = f'ping -c 1 -p {hex_message[:16]} {ip}'
        os.system(command)

    def receive_message(interface):
        try:
            command = f'sudo tcpdump -i {interface} icmp -X -c 1'
            result = subprocess.check_output(command, shell=True).decode()

            print("tcpdump output:")
            print(result)  # Debug: Print the raw tcpdump output

            # Extract the hex payload from the tcpdump output
            hex_message = ''
            payload_started = False
            for line in result.splitlines():
                if payload_started:
                    parts = line.split()
                    if len(parts) > 1:
                        hex_message += ''.join(parts[1:])  # Skip the first part which is the address
                if 'ICMP echo request' in line or 'ICMP echo reply' in line:
                    payload_started = True

            if hex_message:
                return hex_to_text(hex_message)
            else:
                return "No valid ICMP payload found."
        except subprocess.CalledProcessError as e:
            return f"Error running tcpdump: {e}"

    def main():
        while True:
            # Read JSON input
            input_json = input("Enter the input in JSON format: ")
            try:
                data = json.loads(input_json)
            except json.JSONDecodeError:
                print(json.dumps({"status": "error", "message": "Invalid JSON input"}))
                continue
            
            choice = data.get("choice", "").strip().lower()
            if choice == "send":
                ip = data.get("ip", "").strip()
                message = data.get("message", "").strip()
                send_message(ip, message)
                print(json.dumps({"status": "success", "message": f"Message sent to {ip}"}))

            elif choice == "receive":
                interface = data.get("interface", "").strip()
                print("Listening for incoming messages...")
                message = receive_message(interface)
                print(json.dumps({"status": "success", "received_message": message}))

                reply = data.get("reply", "").strip()
                if reply:
                    sender_ip = data.get("sender_ip", "").strip()
                    send_message(sender_ip, reply)
                    print(json.dumps({"status": "success", "message": f"Reply sent to {sender_ip}"}))

            elif choice == "exit":
                print(json.dumps({"status": "success", "message": "Exiting the program."}))
                break

            else:
                print(json.dumps({"status": "error", "message": "Invalid choice. Please choose 'send', 'receive', or 'exit'."}))

if __name__ == "__main__":
    chatPing()

 #docer file allow to run python code   
@app.route('/run_docker', methods=['POST'])
def run_docker():
    data = request.json
    image_name = data.get('image')
    script_path = data.get('script')

    try:
        # Run the Docker command to execute the Python script
        result = subprocess.run(
            ['docker', 'run', '--rm', '-v', f'{script_path}:/app/script.py', image_name, 'python', '/app/script.py'],
            capture_output=True, text=True
        )

        # Check for errors in the Docker command execution
        if result.returncode == 0:
            return jsonify({"status": "success", "output": result.stdout})
        else:
            return jsonify({"status": "error", "error": result.stderr}), 500

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500
    
    
# Set up AI21 Client
client = AI21Client(api_key="37KFsDD0G6VmvHgaNJ1rewV5S8teqCdr")

@app.route('/chat_ai21', methods=['POST'])
def chat_ai21():
    """Handle chat messages sent to AI21 API."""
    user_message = request.json.get('message')

    # Prepare messages to send to AI21 API
    messages = [ChatMessage(role="user", content=user_message)]

    try:
        # Send message to AI21 API
        response = client.chat.completions.create(
            model="jamba-instruct-preview",
            messages=messages,
            top_p=1.0  # Setting to 1 encourages different responses each call.
        )

        # Extract the response content from AI21
        bot_message = response.to_json().get('messages')[0]['content']

        return jsonify({'response': bot_message})

    except Exception as e:
        print(f"Error communicating with AI21 API: {e}")
        return jsonify({'response': 'Sorry, something went wrong with the AI21 service.'}), 500

if __name__ == "__main__":
    app.run(port=80, host="0.0.0.0")