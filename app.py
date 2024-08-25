from flask import Flask, request, jsonify
import smtplib
import ssl
import requests
from bs4 import BeautifulSoup
import geocoder
import pyttsx3
import pywhatkit
import pyautogui
import time
from twilio.rest import Client

app = Flask(__name__)

def send_email(sender_email, receiver_email, password, subject, body):
    try:
        port = 465
        smtp_server = "smtp.gmail.com"
        message = f"""\
Subject: {subject}

{body}"""
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message)
        return "Email sent successfully."
    except Exception as e:
        return f"Failed to send email: {e}"

@app.route('/send_email', methods=['POST'])
def api_send_email():
    data = request.json
    result = send_email(data['sender_email'], data['receiver_email'], data['password'], data['subject'], data['body'])
    return jsonify({'message': result})

def send_sms_via_twilio(to_number, message_body):
    try:
        ACCOUNT_SID = 'YOUR_TWILIO_ACCOUNT_SID'
        AUTH_TOKEN = 'YOUR_TWILIO_AUTH_TOKEN'
        FROM_NUMBER = 'YOUR_TWILIO_PHONE_NUMBER'
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        message = client.messages.create(
            body=message_body,
            from_=FROM_NUMBER,
            to=to_number
        )
        return "SMS sent successfully."
    except Exception as e:
        return f"Failed to send SMS: {e}"

@app.route('/send_sms_twilio', methods=['POST'])
def api_send_sms_twilio():
    data = request.json
    result = send_sms_via_twilio(data['to_number'], data['message_body'])
    return jsonify({'message': result})

def scrape_google(query):
    try:
        search_url = f"https://www.google.com/search?q={query}"
        response = requests.get(search_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        search_results = []
        for result in soup.find_all('h3')[:5]:
            search_results.append(result.get_text())
        return {"results": search_results}
    except Exception as e:
        return {"error": f"Failed to scrape Google search results: {e}"}

@app.route('/scrape_google', methods=['GET'])
def api_scrape_google():
    query = request.args.get('query')
    result = scrape_google(query)
    return jsonify(result)

def get_geo_coordinates():
    try:
        g = geocoder.ip('me')
        return {"location": f"{g.city}, {g.state}, {g.country}", "coordinates": g.latlng}
    except Exception as e:
        return {"error": f"Failed to get geo-coordinates: {e}"}

@app.route('/geo_coordinates', methods=['GET'])
def api_geo_coordinates():
    result = get_geo_coordinates()
    return jsonify(result)

def text_to_audio(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        return "Text converted to audio successfully."
    except Exception as e:
        return f"Failed to convert text to audio: {e}"

@app.route('/text_to_audio', methods=['POST'])
def api_text_to_audio():
    data = request.json
    result = text_to_audio(data['text'])
    return jsonify({'message': result})

def control_volume(volume):
    try:
        pyautogui.press('volumedown', presses=50)
        pyautogui.press('volumeup', presses=volume // 2)
        return f"Volume set to {volume}%"
    except Exception as e:
        return f"Failed to control volume: {e}"

@app.route('/control_volume', methods=['POST'])
def api_control_volume():
    data = request.json
    result = control_volume(data['volume'])
    return jsonify({'message': result})

def send_sms_via_mobile(phone_number, message):
    try:
        pywhatkit.sendwhatmsg_instantly(phone_number, message)
        return f"SMS sent to {phone_number} successfully."
    except Exception as e:
        return f"Failed to send SMS via mobile: {e}"

@app.route('/send_sms_mobile', methods=['POST'])
def api_send_sms_mobile():
    data = request.json
    result = send_sms_via_mobile(data['phone_number'], data['message'])
    return jsonify({'message': result})

def send_bulk_email(sender_email, password, receiver_emails, subject, body):
    results = []
    for receiver_email in receiver_emails:
        result = send_email(sender_email, receiver_email, password, subject, body)
        results.append(result)
        time.sleep(1)
    return results

@app.route('/send_bulk_email', methods=['POST'])
def api_send_bulk_email():
    data = request.json
    results = send_bulk_email(data['sender_email'], data['password'], data['receiver_emails'], data['subject'], data['body'])
    return jsonify({'messages': results})

if __name__ == "__main__":
    app.run(debug=True)
