#!/usr/bin/env python3
"""
Gradio ISL Demo - Railway Deployment Ready
Secure deployment with environment variables for sensitive data
"""

import os
import gradio as gr
import torch
import numpy as np
import cv2
import collections
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import google.generativeai as genai
from dataclasses import dataclass

# Suppress MediaPipe verbose logging
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from common import GRUClassifier, features_from_frame, _import_mediapipe


@dataclass
class PredictedWord:
    """Represents a predicted word with metadata"""
    text: str
    confidence: float
    timestamp: datetime
    selected: bool = True


class WordBuffer:
    """Manages accumulated words with confidence filtering"""
    
    def __init__(self, max_words: int = 10, confidence_threshold: float = 0.6):
        self.max_words = max_words
        self.confidence_threshold = confidence_threshold
        self.words: List[PredictedWord] = []
        self.last_word = ""
        self.last_add_time = datetime.now()
        
    def add_word(self, word: str, confidence: float, min_time_between: float = 2.0) -> bool:
        """Add word if it meets criteria"""
        now = datetime.now()
        
        # Skip if same word added recently
        if (word == self.last_word and 
            (now - self.last_add_time).total_seconds() < min_time_between):
            return False
            
        # Skip if confidence too low
        if confidence < self.confidence_threshold:
            return False
            
        # Skip if word already exists (prevent duplicates)
        existing_words = [w.text for w in self.words]
        if word in existing_words:
            return False
            
        # Remove oldest if buffer full
        if len(self.words) >= self.max_words:
            self.words.pop(0)
        
        # Add new word
        new_word = PredictedWord(
            text=word,
            confidence=confidence,
            timestamp=now
        )
        
        self.words.append(new_word)
        self.last_word = word
        self.last_add_time = now
        
        return True
        
    def get_words_text(self) -> List[str]:
        """Get list of word texts"""
        return [w.text for w in self.words]
        
    def get_words_display(self) -> str:
        """Get formatted string for display"""
        if not self.words:
            return "No words detected yet"
        return " | ".join([f"{w.text} ({w.confidence:.2f})" for w in self.words])
        
    def clear(self):
        """Clear all words"""
        self.words.clear()
        self.last_word = ""


class GeminiSentencePredictor:
    """Handles Gemini API integration for sentence prediction"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        if api_key and api_key != "your_api_key_here":
            try:
                genai.configure(api_key=api_key)
                self.gen_model = genai.GenerativeModel('gemini-1.5-flash')
                self.enabled = True
                print("✅ Gemini API initialized successfully")
            except Exception as e:
                print(f"⚠️ Gemini API initialization failed: {e}")
                self.enabled = False
        else:
            print("⚠️ No Gemini API key provided - using fallback only")
            self.enabled = False
        
        self.sentence_history: List[str] = []
        
    def predict_sentence(self, words: List[str]) -> str:
        """Predict sentence from list of words using Gemini or fallback"""
        if not words:
            return ""
            
        # Try Gemini first if available
        if self.enabled:
            try:
                prompt = f"""You are an expert Indian Sign Language (ISL) interpreter helping deaf users communicate naturally.

I will give you words that were signed in ISL. Your job is to create a natural, fluent English sentence that captures what the person was trying to communicate.

ISL CONTEXT:
- ISL users think and express ideas naturally, just like spoken language users
- You can add any English words (articles, prepositions, verbs, etc.) needed for fluent communication
- Focus on natural meaning, not literal word-for-word translation
- Make it sound like something a person would actually say in conversation

SIGNED WORDS: {', '.join(words)}

Create a natural English sentence that expresses what the signer meant to communicate. Be conversational and natural.

English sentence:"""

                response = self.gen_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=60,
                        temperature=0.7,
                        top_p=0.9
                    )
                )
                
                if response and hasattr(response, 'text') and response.text:
                    sentence = response.text.strip()
                    # Clean up response formatting
                    if sentence.startswith("English sentence:"):
                        sentence = sentence[17:].strip()
                    if sentence.startswith("-"):
                        sentence = sentence[1:].strip()
                    if sentence.startswith('"') and sentence.endswith('"'):
                        sentence = sentence[1:-1]
                    
                    # Store in history
                    if sentence and sentence not in self.sentence_history:
                        self.sentence_history.append(sentence)
                        if len(self.sentence_history) > 5:
                            self.sentence_history.pop(0)
                            
                    return sentence
                    
            except Exception as e:
                print(f"Gemini API error: {e}")
        
        # Fallback to rule-based sentence formation
        return self._create_natural_fallback_sentence(words)
            
    def _create_natural_fallback_sentence(self, words: List[str]) -> str:
        """Create a natural conversational sentence as fallback"""
        if not words:
            return ""
        
        # Convert to lowercase for processing
        lower_words = [w.lower() for w in words]
        
        # Natural sentence patterns based on common ISL communication
        if len(words) == 1:
            word = lower_words[0]
            if word in ['happy', 'sad', 'angry', 'tired', 'hungry', 'thirsty']:
                return f"I am {word}."
            elif word in ['camera', 'phone', 'computer']:
                return f"I need the {word}."
            elif word in ['food', 'water', 'help']:
                return f"I want {word}."
            elif word in ['home', 'school', 'work']:
                return f"I am going to {word}."
            else:
                return f"I see a {word}."
                
        elif len(words) == 2:
            w1, w2 = lower_words[0], lower_words[1]
            
            # Common two-word combinations
            if w1 == 'happy' and w2 == 'birthday':
                return "Happy birthday!"
            elif w1 in ['me', 'i'] and w2 in ['hungry', 'tired', 'happy']:
                return f"I am {w2}."
            elif w1 in ['go', 'come'] and w2 in ['home', 'school', 'work']:
                return f"I am going {w2}."
            elif w1 == 'camera' and w2 in ['photo', 'picture']:
                return "I want to take a photo with the camera."
            elif w1 in ['poster', 'picture'] and w2 == 'animals':
                return f"I see a {w1} with {w2}."
            else:
                return f"I see the {w1} and {w2}."
                
        elif len(words) == 3:
            w1, w2, w3 = lower_words[0], lower_words[1], lower_words[2]
            
            # Three-word natural patterns
            if 'camera' in lower_words and 'poster' in lower_words and 'animals' in lower_words:
                return "I took a photo of the animal poster."
            elif 'tomorrow' in lower_words and 'school' in lower_words:
                return "Tomorrow I will go to school."
            elif 'me' in lower_words or 'i' in lower_words:
                other_words = [w for w in lower_words if w not in ['me', 'i']]
                if len(other_words) == 2:
                    return f"I {other_words[0]} {other_words[1]}."
            elif 'exercise' in lower_words and 'healthy' in lower_words:
                return "Exercise keeps me healthy."
            else:
                return f"I see {w1}, {w2}, and {w3}."
        else:
            # For more words, create natural groupings
            if len(words) == 4:
                return f"I see {lower_words[0]}, {lower_words[1]}, {lower_words[2]}, and {lower_words[3]}."
            else:
                return f"There are many things: {', '.join(lower_words[:3])} and more."


class ISLPredictor:
    """Main ISL prediction class for Gradio interface"""
    
    def __init__(self):
        print("🚀 Initializing ISL Predictor...")
        
        # Get configuration from environment variables
        self.gemini_api_key = os.getenv('GEMINI_API_KEY', '')
        self.model_path = os.getenv('MODEL_PATH', 'checkpoints/best_gru.pt')
        
        # Initialize components
        self.setup_model()
        self.setup_gemini()
        self.setup_mediapipe()
        
        # Initialize word buffer
        self.word_buffer = WordBuffer(max_words=10, confidence_threshold=0.6)
        
        print("✅ ISL Predictor initialized successfully!")
        
    def setup_model(self):
        """Initialize the GRU model"""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading model on {device}...")
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
                
            ckpt = torch.load(self.model_path, map_location=device, weights_only=False)
            self.class_to_idx = ckpt["class_to_idx"]
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
            
            # Use model configuration from checkpoint if available
            if "args" in ckpt:
                model_args = ckpt["args"]
                hidden = model_args.hidden
                layers = model_args.layers 
                dropout = model_args.dropout
            else:
                hidden = 256
                layers = 2
                dropout = 0.3
                
            self.model = GRUClassifier(
                in_dim=150, hid=hidden, num_layers=layers,
                num_classes=len(self.class_to_idx), dropout=dropout, bidir=True
            ).to(device)
            self.model.load_state_dict(ckpt["model"])
            self.model.eval()
            self.device = device
            
            print(f"✅ Model loaded successfully with {len(self.class_to_idx)} classes")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def setup_gemini(self):
        """Initialize Gemini predictor"""
        self.gemini = GeminiSentencePredictor(self.gemini_api_key)
        
    def setup_mediapipe(self):
        """Initialize MediaPipe"""
        try:
            mp = _import_mediapipe()
            self.holistic = mp.solutions.holistic.Holistic(
                model_complexity=0,  # Lightest model for deployment
                smooth_landmarks=True, 
                enable_segmentation=False,
                refine_face_landmarks=False, 
                min_detection_confidence=0.5, 
                min_tracking_confidence=0.5
            )
            print("✅ MediaPipe initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing MediaPipe: {e}")
            raise
    
    def process_video_frame(self, frame):
        """Process a single video frame and return prediction"""
        try:
            if frame is None:
                return None, 0.0
                
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.holistic.process(rgb_frame)
            
            # Extract features
            features = features_from_frame(results)
            
            # Predict with model
            x = torch.from_numpy(features[None, None, ...]).to(self.device)  # (1, 1, 150)
            
            with torch.no_grad():
                logits = self.model(x)
                prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred_idx = int(prob.argmax())
                confidence = float(prob[pred_idx])
                
            predicted_word = self.idx_to_class[pred_idx]
            
            return predicted_word, confidence
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None, 0.0
    
    def predict_from_video(self, video_path):
        """Process video and return results"""
        if video_path is None:
            return "No video provided", "", ""
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return "Error: Could not open video", "", ""
            
            predictions = []
            frame_count = 0
            
            # Process video frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 5th frame for efficiency
                if frame_count % 5 == 0:
                    word, confidence = self.process_video_frame(frame)
                    if word and confidence > 0.5:
                        if self.word_buffer.add_word(word, confidence):
                            predictions.append(f"{word} ({confidence:.2f})")
                
                frame_count += 1
                
                # Limit processing to avoid timeout
                if frame_count > 150:  # ~5 seconds at 30fps
                    break
            
            cap.release()
            
            # Get current words and create sentence
            words = self.word_buffer.get_words_text()
            words_display = self.word_buffer.get_words_display()
            
            if words:
                sentence = self.gemini.predict_sentence(words)
                return sentence, words_display, f"Processed {frame_count} frames"
            else:
                return "No clear signs detected. Please sign clearly towards the camera.", "", f"Processed {frame_count} frames"
                
        except Exception as e:
            return f"Error processing video: {str(e)}", "", ""
    
    def clear_words(self):
        """Clear accumulated words"""
        self.word_buffer.clear()
        return "Words cleared", "", ""


# Initialize the predictor globally
print("🚀 Starting ISL Demo...")
try:
    predictor = ISLPredictor()
    print("✅ ISL Demo ready!")
except Exception as e:
    print(f"❌ Failed to initialize ISL Demo: {e}")
    predictor = None


def process_video_interface(video):
    """Interface function for Gradio"""
    if predictor is None:
        return "❌ Model not loaded. Please check deployment configuration.", "", ""
    
    if video is None:
        return "Please upload a video or record one using your camera.", "", ""
    
    return predictor.predict_from_video(video)


def clear_interface():
    """Clear interface function for Gradio"""
    if predictor is None:
        return "❌ Model not loaded.", "", ""
    
    return predictor.clear_words()


# Create Gradio interface
with gr.Blocks(title="ISL to English Translation", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🤟 Indian Sign Language to English Translation
        
        Upload a video or record yourself signing in ISL, and get natural English sentences!
        
        **How to use:**
        1. Click on the video input below
        2. Record yourself signing or upload a video file
        3. Click "Translate Signs" to get your English sentence
        4. Use "Clear Words" to start fresh
        """
    )
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(
                label="Sign Language Video",
                sources=["webcam", "upload"]
            )
            
            with gr.Row():
                translate_btn = gr.Button("🔄 Translate Signs", variant="primary")
                clear_btn = gr.Button("🗑️ Clear Words", variant="secondary")
        
        with gr.Column():
            sentence_output = gr.Textbox(
                label="📝 English Sentence",
                placeholder="Your translated sentence will appear here...",
                lines=3
            )
            
            words_output = gr.Textbox(
                label="🔤 Detected Words",
                placeholder="Individual words detected will be shown here...",
                lines=2
            )
            
            status_output = gr.Textbox(
                label="ℹ️ Status",
                placeholder="Processing status...",
                lines=1
            )
    
    # Button actions
    translate_btn.click(
        fn=process_video_interface,
        inputs=[video_input],
        outputs=[sentence_output, words_output, status_output]
    )
    
    clear_btn.click(
        fn=clear_interface,
        inputs=[],
        outputs=[sentence_output, words_output, status_output]
    )
    
    gr.Markdown(
        """
        ### 💡 Tips for better results:
        - Sign clearly and at a moderate pace
        - Ensure good lighting and clear background
        - Keep your hands visible in the camera frame
        - Sign one word at a time with brief pauses
        
        ### 🔧 Technical Details:
        - Uses MediaPipe for pose detection
        - GRU neural network for sign classification
        - Gemini AI for natural sentence formation
        - Supports real-time video processing
        """
    )


if __name__ == "__main__":
    # Get port from environment (Railway sets this automatically)
    port = int(os.getenv("PORT", 7860))
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,  # Don't create public links in deployment
        show_error=True
    )