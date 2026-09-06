import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[1].id)  # Zira (female voice) explicitly set
engine.setProperty("rate", 170)
engine.setProperty("volume", 1.0)  # volume max
engine.say("Assalam o Alaikum, This is testing")
engine.runAndWait()
print("Done speaking")