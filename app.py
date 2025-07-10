from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from gtts import gTTS
import whisper
import os
import requests



def download_small_model():
    model_path = os.path.expanduser("~/.cache/whisper/small.pt")
    if not os.path.exists(model_path):
        print("📦 Downloading small Whisper model from Google Drive...")
        url = "https://drive.google.com/uc?export=download&id=1UXX6-b5PC1sbpky8YoaVfCVO4hjyvbDH"
        response = requests.get(url)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            f.write(response.content)
        print("✅ small Whisper model downloaded!")
    else:
        print("🟢 small Whisper model already exists.")

download_small_model()



app = Flask(__name__)
CORS(app)

# Load Whisper model once at startup
model = whisper.load_model("small")

@app.route('/')
def home():
    return "🎙️ Voice Assistant Backend is Live!"

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    path = "temp.wav"
    audio_file.save(path)

    result = model.transcribe(path)
    os.remove(path)
    return jsonify({'text': result['text']})

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    tts = gTTS(text=text, lang='en', slow=False)
    output_path = "speech.mp3"
    tts.save(output_path)

    return send_file(output_path, mimetype='audio/mpeg')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    # Step 1: Audio input → transcribe
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    path = "temp_chat.wav"
    audio_file.save(path)

    # Step 2: Transcribe with Whisper
    result = model.transcribe(path)
    user_text = result['text']
    os.remove(path)

    # Step 3: Generate a reply (simple version, rule-based)
    reply = get_bot_reply(user_text)

    # Step 4: Convert reply to speech
    tts = gTTS(text=reply, lang='en', slow=False)
    reply_path = "reply.mp3"
    tts.save(reply_path)

    return send_file(reply_path, mimetype='audio/mpeg')

#  Simple rule-based assistant logic
def get_bot_reply(user_input):
    user_input = user_input.lower()

    if "hello" in user_input:
        return "Hello babe, how can I help you today?"
    elif "time" in user_input:
        from datetime import datetime
        return f"The time is {datetime.now().strftime('%I:%M %p')}."
    elif "your name" in user_input:
        return "I’m your sweet voice assistant!"
    else:
        return "Sorry, I didn’t understand that. Try again?"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
