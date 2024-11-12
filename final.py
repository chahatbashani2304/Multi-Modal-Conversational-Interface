import speech_recognition as sr
import os
import nltk
import random
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from nltk.stem import WordNetLemmatizer
import string
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import tkinter as tk
from tkinter import filedialog, messagebox

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Load your intents JSON file
with open("C:\\Users\\CHAHAT BASHANI\\OneDrive\\Documents\\intents.json", 'r', encoding='utf-8') as file:

    data = json.load(file)

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Initialize the BLIP model for image captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Preprocessing for chatbot
words = []
classes = []
data_x = []
data_y = []

lemmatizer = WordNetLemmatizer()

# Tokenize and preprocess
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)  # Tokenizing patterns
        words.extend(tokens)
        data_x.append(pattern)
        data_y.append(intent["tag"])

# Lemmatize and sort words and classes
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
words = sorted(set(words))
classes = sorted(set(data_y))

# Creating bag of words model
training = []
out_empty = [0] * len(classes)

for idx, doc in enumerate(data_x):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)

    output_row = list(out_empty)
    output_row[classes.index(data_y[idx])] = 1

    training.append([bow, output_row])

# Shuffle training data and convert to numpy array
random.shuffle(training)
training = np.array(training, dtype=object)

train_X = np.array(list(training[:, 0]))
train_Y = np.array(list(training[:, 1]))

# Neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_X[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_Y[0]), activation="softmax"))

adam = tf.keras.optimizers.Adam(learning_rate=0.01)

def model_compile(loss, optimizer, metrics):
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

model_compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Train the model
model.fit(x=train_X, y=train_Y, epochs=150, verbose=1)

# Function to predict the class of the input message
def pred_class(message, words, classes):
    # Tokenize and lemmatize the input message
    message_words = nltk.word_tokenize(message)
    message_words = [lemmatizer.lemmatize(w.lower()) for w in message_words]

    # Create a bag of words for the message
    bow = [0] * len(words)
    for s in message_words:
        if s in words:
            bow[words.index(s)] = 1

    # Predict the intent
    pred = model.predict(np.array([bow]))  # Predict using the trained model
    return pred

# Function to get the chatbot's response
def get_response(intents_list, data):
    tag = intents_list[0]['intent']  # Extract the intent tag
    list_of_intents = data['intents']
    result = ""

    for intent in list_of_intents:
        if intent['tag'] == tag:
            result = random.choice(intent['responses'])  # Randomly select a response
            break
    return result

# Function to recognize speech from an uploaded audio file
def recognize_audio(file_path):
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    try:
        # Recognize the audio using Google Web Speech API
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, I could not understand the audio."
    except sr.RequestError:
        return "Sorry, there was an error with the speech recognition service."

# GUI for Tkinter
class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ChatBot with Tkinter")

        # Set up the layout
        self.chat_display = tk.Text(root, height=20, width=60, wrap=tk.WORD)
        self.chat_display.pack(padx=10, pady=10)

        self.input_entry = tk.Entry(root, width=50)
        self.input_entry.pack(padx=10, pady=10)

        self.send_button = tk.Button(root, text="Send", width=10, command=self.send_message)
        self.send_button.pack(pady=5)

        self.audio_button = tk.Button(root, text="Upload Audio", width=10, command=self.upload_audio)
        self.audio_button.pack(pady=5)

        self.image_button = tk.Button(root, text="Upload Image", width=10, command=self.upload_image)
        self.image_button.pack(pady=5)

        # Quit button to close the application at the bottom-right corner
        self.quit_button = tk.Button(root, text="Quit", width=10, command=self.quit_app)
        self.quit_button.place(relx=1.0, rely=1.0, anchor='se', x=-10, y=-10)

        # Handling window close event
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

    def send_message(self):
        user_message = self.input_entry.get()
        self.chat_display.insert(tk.END, f"You: {user_message}\n")
        self.input_entry.delete(0, tk.END)

        intents = pred_class(user_message, words, classes)
        intent = {"intent": classes[np.argmax(intents)]}
        response = get_response([intent], data)
        self.chat_display.insert(tk.END, f"Bot: {response}\n\n")  # Added blank line after bot's response

    def upload_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
        if file_path:
            text_from_audio = recognize_audio(file_path)
            self.chat_display.insert(tk.END, f"You (Audio): {text_from_audio}\n")

            intents = pred_class(text_from_audio, words, classes)
            intent = {"intent": classes[np.argmax(intents)]}
            response = get_response([intent], data)
            self.chat_display.insert(tk.END, f"Bot: {response}\n\n")  # Added blank line after bot's response

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            raw_image = Image.open(file_path).convert("RGB")
            inputs = blip_processor(raw_image, return_tensors="pt")
            out = blip_model.generate(**inputs)
            caption = blip_processor.decode(out[0], skip_special_tokens=True)
            self.chat_display.insert(tk.END, f"Image Caption: {caption}\n\n")  # Added blank line after caption

    def quit_app(self):
        self.root.quit()  # Ends the Tkinter event loop and closes the application
        self.root.destroy()  # Destroys the main window

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()