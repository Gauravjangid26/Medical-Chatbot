import pyttsx3
def speak_response(text):
    engine = pyttsx3.init(driverName='nsss')
    engine.say(text)
    engine.runAndWait()
    
speak_response('my name is gaurav')