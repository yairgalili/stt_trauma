import whisper

model = whisper.load_model("large")

# load audio and pad/trim it to fit 30 seconds
audio_path = r"data\a.m4a"

result = model.transcribe(audio_path, fp16=False,language='Hebrew',verbose=True)
print(result)