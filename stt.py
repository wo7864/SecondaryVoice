import speech_recognition as sr

def stt(audio):
    r = sr.Recognizer()
    if type(audio) == str:
        audio = sr.AudioFile(audio)
    with audio as source:
        audio = r.record(source)
    
    return r.recognize_google(audio)

if __name__ == "__main__":
    audio_path = 'a001.wav'
    stt(audio_path)