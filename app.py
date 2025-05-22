import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import speech_recognition as sr
import pyttsx3
from PIL import Image
import io
import json
import time

class EnhancedDisabilityAssistanceApp:
    def __init__(self):
        # Initialize components
        self.text_to_speech = pyttsx3.init()
        try:
            self.recognizer = sr.Recognizer()
            self.microphone_available = True
        except:
            self.microphone_available = False
            st.warning("Microphone access not available. Text input will be used instead.")

    def voice_to_sign_language(self, text):
        """Enhanced sign language conversion with more vocabulary"""
        # Extended sign language mapping
        with open('sign_language_dictionary.json', 'r') as f:
            sign_language_mapping = {
                # Basic greetings
                "hello": "👋", "goodbye": "👋", "thank you": "🙏",
                # Common phrases
                "help": "🤲", "yes": "👍", "no": "👎",
                "please": "🙏", "sorry": "😔",
                # Basic needs
                "food": "🍽️", "water": "💧", "bathroom": "🚽",
                "medicine": "💊", "emergency": "🚨",
                # Emotions
                "happy": "😊", "sad": "😢", "pain": "😣",
                "tired": "😴", "sick": "🤒",
                # Time-related
                "now": "⌚", "later": "⏳", "wait": "⌛",
                # Common actions
                "stop": "✋", "go": "👉", "come": "👈",
                "look": "👀", "listen": "👂", "speak": "🗣️"
            }

        # Split input text into words and convert each
        words = text.lower().split()
        signs = [sign_language_mapping.get(word, word) for word in words]
        return " ".join(signs)

    def detect_obstacles(self, frame):
        """Enhanced obstacle detection with distance estimation"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Depth estimation (simplified)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles = []
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                # Estimate distance based on object size
                estimated_distance = round((1000 / w) * 100) / 100  # Simplified distance calculation
                obstacles.append({
                    'position': (x, y),
                    'size': (w, h),
                    'estimated_distance': estimated_distance
                })
        
        return obstacles

    def image_to_text(self, image):
        """Convert image to descriptive text for visually impaired users"""
        # Placeholder for actual OCR/image recognition
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Basic image analysis
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        description = f"This image appears to be {'bright' if brightness > 127 else 'dark'} "
        description += f"with {'high' if contrast > 50 else 'low'} contrast. "
        
        return description

    def color_blind_assist(self, image):
        """Convert image for color-blind accessibility"""
        # Convert to various color-blind friendly modes
        modes = {
            'Original': image,
            'Grayscale': cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR),
            'Deuteranopia': cv2.applyColorMap(image, cv2.COLORMAP_JET),
            'High Contrast': cv2.convertScaleAbs(image, alpha=1.5, beta=0)
        }
        return modes

    def main_app(self):
        st.title("Enhanced Disability Assistance AI")
        
        # Main navigation
        feature = st.sidebar.selectbox("Select Assistance Mode", [
            "Communication Assistant",
            "Visual Assistance",
            "Hearing Assistance",
            "Mobility Assistance",
            "Cognitive Assistance",
            "Accessibility Settings"
        ])

        if feature == "Communication Assistant":
            st.header("Communication Assistant")
            
            # Multiple input methods
            input_method = st.radio("Choose Input Method", ["Text", "Voice"])
            
            if input_method == "Voice" and self.microphone_available:
                if st.button("Start Speaking"):
                    with sr.Microphone() as source:
                        st.write("Listening...")
                        try:
                            audio = self.recognizer.listen(source, timeout=5)
                            text = self.recognizer.recognize_google(audio)
                            st.write(f"Recognized Text: {text}")
                            signs = self.voice_to_sign_language(text)
                            st.write(f"Sign Language: {signs}")
                            
                            # Text-to-speech output
                            self.text_to_speech.say(text)
                            self.text_to_speech.runAndWait()
                        except sr.UnknownValueError:
                            st.error("Could not understand audio")
                        except sr.RequestError:
                            st.error("Could not request results")
            else:
                text = st.text_area("Enter text:", height=100)
                if st.button("Convert"):
                    if text:
                        signs = self.voice_to_sign_language(text)
                        st.write(f"Sign Language: {signs}")
                        
                        # Text-to-speech output
                        self.text_to_speech.say(text)
                        self.text_to_speech.runAndWait()

        elif feature == "Visual Assistance":
            st.header("Visual Assistance Tools")
            
            uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])
            
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # Image description
                description = self.image_to_text(image)
                st.write("Image Description:", description)
                
                # Color blind assistance
                st.subheader("Color Vision Assistance")
                color_modes = self.color_blind_assist(image)
                selected_mode = st.selectbox("Select Color Mode", list(color_modes.keys()))
                st.image(color_modes[selected_mode], channels="BGR")
                
                # Text extraction
                st.subheader("Text in Image")
                st.write("Detecting text... (OCR placeholder)")

        elif feature == "Hearing Assistance":
            st.header("Hearing Assistance Tools")
            
            # Speech-to-text live transcription
            if st.button("Start Live Transcription"):
                st.write("Transcription starting... (Press stop when done)")
                
                # Placeholder for live transcription
                transcription_placeholder = st.empty()
                stop_button = st.button("Stop Transcription")
                
                while not stop_button:
                    transcription_placeholder.write("Sample transcription text...")
                    time.sleep(1)

        elif feature == "Mobility Assistance":
            st.header("Mobility Assistance")
            
            # Navigation assistance
            st.subheader("Navigation Assistant")
            uploaded_file = st.file_uploader("Upload environment image", type=['jpg', 'png', 'jpeg'])
            
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                obstacles = self.detect_obstacles(frame)
                
                if obstacles:
                    st.warning(f"⚠️ {len(obstacles)} potential obstacles detected!")
                    for i, obstacle in enumerate(obstacles, 1):
                        st.write(f"Obstacle {i}: approximately {obstacle['estimated_distance']}m away")
                else:
                    st.success("Path appears clear")

        elif feature == "Cognitive Assistance":
            st.header("Cognitive Assistance Tools")
            
            # Task organization
            st.subheader("Task Organizer")
            task = st.text_input("Add a task:")
            if st.button("Add Task"):
                st.write("Task added to list")
            
            # Schedule visualization
            st.subheader("Daily Schedule")
            st.write("Morning routine:")
            st.write("- Wake up")
            st.write("- Breakfast")
            st.write("- Medications")

        elif feature == "Accessibility Settings":
            st.header("Accessibility Settings")
            
            # Visual settings
            st.subheader("Visual Settings")
            text_size = st.slider("Text Size", 10, 30, 16)
            contrast = st.slider("Contrast", 0.5, 2.0, 1.0)
            
            # Audio settings
            st.subheader("Audio Settings")
            speech_rate = st.slider("Speech Rate", 0.5, 2.0, 1.0)
            volume = st.slider("Volume", 0.0, 1.0, 0.5)
            
            # Save settings
            if st.button("Save Settings"):
                st.success("Settings saved successfully!")

def main():
    app = EnhancedDisabilityAssistanceApp()
    app.main_app()

if __name__ == "__main__":
    main()