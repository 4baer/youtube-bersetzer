import os
import logging
from flask import Flask, request, jsonify, send_from_directory
from pytube import YouTube
import moviepy.editor as mp
import whisper
from transformers import MarianMTModel, MarianTokenizer
from gtts import gTTS
from pydub import AudioSegment
import threading

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure temporary directory exists
TEMP_DIR = "temp_audio"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# Ensure static directory exists
STATIC_DIR = "static"
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# Global variable to track progress
progress = {
    "percentage": 0,
    "message": "",
    "video_path": None
}

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/process', methods=['POST'])
def process_video():
    data = request.json
    url = data['url']
    
    def process():
        global progress
        try:
            logger.info(f"Processing URL: {url}")

            # Step 1: Download YouTube video
            yt = YouTube(url)
            video_stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
            if not video_stream:
                raise ValueError("No valid video stream found.")
            video_path = video_stream.download(filename='video.mp4')
            logger.info(f"Video downloaded to {video_path}")
            progress = {"percentage": 20, "message": "Video downloaded"}

            # Step 2: Extract audio from the video
            video_clip = mp.VideoFileClip(video_path)
            audio_path = os.path.join(TEMP_DIR, 'audio.wav')
            video_clip.audio.write_audiofile(audio_path)
            logger.info(f"Audio extracted to {audio_path}")
            progress = {"percentage": 40, "message": "Audio extracted"}

            # Step 3: Generate German audio
            german_audio_path = generate_german_audio(audio_path)
            logger.info(f"German audio generated at {german_audio_path}")
            progress = {"percentage": 60, "message": "German audio generated"}

            # Step 4: Replace audio in video
            output_video_path = os.path.join(STATIC_DIR, 'output_video.mp4')
            replace_audio_in_video(video_path, german_audio_path, output_video_path)
            logger.info(f"Output video saved at {output_video_path}")
            progress = {"percentage": 100, "message": "Processing complete", "video_path": output_video_path}

        except Exception as e:
            logger.error(f"Error processing video: {e}")
            progress = {"percentage": 0, "message": f"Error: {str(e)}", "video_path": None}

    thread = threading.Thread(target=process)
    thread.start()

    return jsonify({"message": "Processing started"}), 202

@app.route('/progress', methods=['GET'])
def get_progress():
    global progress
    return jsonify(progress)

def generate_german_audio(audio_path):
    # Step 1: Transcribe audio to text with timing information
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    segments = result["segments"]

    # Step 2: Translate each segment to German
    model_name = "Helsinki-NLP/opus-mt-en-de"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    german_audio_segments = []
    
    for segment in segments:
        english_text = segment["text"]
        start_time = segment["start"]
        end_time = segment["end"]
        duration = (end_time - start_time) * 1000  # Convert to milliseconds

        # Translate the segment text
        translated = model.generate(**tokenizer(english_text, return_tensors="pt", padding=True))
        german_text = tokenizer.decode(translated[0], skip_special_tokens=True)

        # Convert translated text to speech
        tts = gTTS(german_text, lang='de')
        segment_audio_path = os.path.join(TEMP_DIR, f'german_segment_{start_time}.mp3')
        tts.save(segment_audio_path)

        # Convert mp3 to wav for compatibility
        sound = AudioSegment.from_mp3(segment_audio_path)
        
        # Adjust the speed of the segment
        sound = sound.speedup(playback_speed=1.5)
        
        # Trim or pad the segment to match the original duration
        if len(sound) > duration:
            sound = sound[:duration]
        else:
            padding = AudioSegment.silent(duration=duration - len(sound))
            sound = sound + padding

        segment_audio_wav_path = os.path.join(TEMP_DIR, f'german_segment_{start_time}.wav')
        sound.export(segment_audio_wav_path, format="wav")

        german_audio_segments.append((start_time, segment_audio_wav_path))

    # Combine all segments into one audio file
    combined = AudioSegment.silent(duration=0)
    for start_time, segment_path in german_audio_segments:
        segment = AudioSegment.from_wav(segment_path)
        combined = combined + segment

    combined_audio_path = os.path.join(TEMP_DIR, 'combined_german_audio.wav')
    combined.export(combined_audio_path, format="wav")

    return combined_audio_path

def replace_audio_in_video(video_path, audio_path, output_path):
    video_clip = mp.VideoFileClip(video_path)
    audio_clip = mp.AudioFileClip(audio_path)
    new_video = video_clip.set_audio(audio_clip)
    new_video.write_videofile(output_path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')