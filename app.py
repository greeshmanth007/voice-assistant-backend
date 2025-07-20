from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from gtts import gTTS
import whisper
import os
from openai import OpenAI

app = Flask(__name__)
CORS(app)

model = whisper.load_model("small")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "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"))

@app.route('/')
def home():
    return "Voice Assistant Backend is Live!"

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    audio_file = request.files['audio']
    path = "temp.wav"
    audio_file.save(path)
    result = model.transcribe(path, fp16=False)
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
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    audio_file = request.files['audio']
    path = "temp_chat.wav"
    audio_file.save(path)
    result = model.transcribe(path, fp16=False)
    user_text = result['text']
    os.remove(path)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a friendly, helpful AI voice assistant."},
            {"role": "user", "content": user_text}
        ]
    )
    reply = completion.choices[0].message.content
    tts = gTTS(text=reply, lang='en', slow=False)
    reply_path = "reply.mp3"
    tts.save(reply_path)
    return send_file(reply_path, mimetype='audio/mpeg')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
