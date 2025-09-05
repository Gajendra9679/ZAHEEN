from flask import Flask, render_template, Response, request, jsonify
import cv2
import time
from deepface import DeepFace
import google.generativeai as genai
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_url_path='/static')

# Configure the API key
API_KEY = "AIzaSyAQ6BNU3nrFTVmU7Dva4tKKrNLNq8ny3cw"
genai.configure(api_key=API_KEY)

# Open webcam
cam = cv2.VideoCapture(0)
max_emotion = "Neutral"
max_count = 0
emotion_history = []
emotion_buffer_size = 5  # Store last 5 emotions for smoothing

def detect_emotion_deepface(frame):
    try:
        # Use DeepFace to analyze emotion
        result = DeepFace.analyze(
            img_path=frame, 
            actions=['emotion'], 
            enforce_detection=False,
            detector_backend='opencv'
        )
        
        # Handle both list and dict results (DeepFace can return either)
        if isinstance(result, list):
            dominant_emotion = result[0]['dominant_emotion']
        else:
            dominant_emotion = result['dominant_emotion']
            
        return dominant_emotion.capitalize()
    except Exception as e:
        logger.error(f"DeepFace Error: {str(e)}")
        return "Neutral"

def smooth_emotion(new_emotion):
    """Apply smoothing to prevent emotion flickering"""
    global emotion_history
    
    # Add new emotion to history
    emotion_history.append(new_emotion)
    
    # Keep history at maximum size
    if len(emotion_history) > emotion_buffer_size:
        emotion_history.pop(0)
    
    # Count frequencies
    emotion_counts = {}
    for emotion in emotion_history:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    # Return most frequent emotion
    return max(emotion_counts, key=emotion_counts.get)

def detection():
    global max_emotion
    last_analysis_time = time.time()
    analysis_interval = 0.5  # Analyze every 0.5 seconds
    
    while True:
        ret, frame = cam.read()
        if not ret:
            logger.error("Failed to capture frame from camera")
            break

        current_time = time.time()
        
        # Don't analyze every frame (performance optimization)
        if current_time - last_analysis_time >= analysis_interval:
            emotion = detect_emotion_deepface(frame)
            
            # Special case handling
            if emotion == "Surprise":
                emotion = "Neutral"  # As per your original logic
                
            # Apply smoothing
            smooth_result = smooth_emotion(emotion)
            max_emotion = smooth_result
            last_analysis_time = current_time

        # Display the emotion on the frame
        cv2.putText(
            frame, 
            f"Emotion: {max_emotion}", 
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.2, 
            (0, 0, 255), 
            2
        )
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Small delay to reduce CPU usage
        time.sleep(0.03)
        
        if cv2.waitKey(1) & 0xFF == 13:  # Enter key
            break

    cam.release()
    cv2.destroyAllWindows()

def bot_answer(question, current_mood):
    """Generate a response from the chatbot using direct Google Generative AI (no LangChain)"""
    logger.info(f"Processing question with mood: {current_mood}")
    
    try:
        # Determine prompt based on mood
        if current_mood in ['Happy', 'Surprise']:
            prompt = f"""You are a Personal Therapist named Zaheen. Your goal is to uplift user mental health and engage in a deep human-like conversation. 
            Your goal as therapist should be to engage in therapeutic conversations, asking follow up questions and talk like a caring friend like a counsellor. 
            The user is in a happy and joyful mood. Acknowledge this and complement them accordingly.
            Keeping that in mind, the query of the user is: {question}
            Respond with a warm, empathetic, joyful tone, and aim to improve their emotional well-being. 
            Keep it engaging by adding a follow up question, like a human conversation. Limit your answer to 60-80 words."""
        elif current_mood in ['Angry', 'Disgust']:
            prompt = f"""You are a Personal Therapist named Zaheen. Your goal is to uplift user mental health and engage in a deep human-like conversation. 
            Your goal as therapist should be to engage in therapeutic conversations, asking follow up questions and talk like a caring friend like a counsellor. 
            The user appears to be in an angry mood. Respond with calm understanding.
            Keeping that in mind, the query of the user is: {question}
            Respond with a warm, empathetic tone, and aim to improve their emotional well-being. 
            Keep a calming tone and try to calm down the user by hearing and replying appropriately.
            Keep it engaging by adding a follow up question, like a human conversation. Limit your answer to 60-80 words."""
        elif current_mood in ['Fear', 'Sad']:
            prompt = f"""You are a Personal Therapist named Zaheen. Your goal is to uplift user mental health and engage in a deep human-like conversation. 
            Your goal as therapist should be to engage in therapeutic conversations, asking follow up questions and talk like a caring friend like a counsellor. 
            The user seems to be in a sad mood. Acknowledge this and console them accordingly.
            Keeping that in mind, the query of the user is: {question}
            Respond with a warm, empathetic tone, and aim to improve their emotional well-being.
            Keep it engaging by adding a follow up question, like a human conversation. Limit your answer to 60-80 words."""
        else:
            prompt = f"""You are a Personal Therapist named Zaheen. Your goal is to uplift user mental health and engage in a deep human-like conversation. 
            Your goal as therapist should be to engage in therapeutic conversations, asking follow up questions and talk like a caring friend like a counsellor. 
            Keeping that in mind, the query of the user is: {question}
            Respond with a warm, empathetic tone, and aim to improve their emotional well-being.
            Keep it engaging by adding a follow up question, like a human conversation. Limit your answer to 60-80 words."""
        
        # Configure the generation parameters
        generation_config = {
            "temperature": 0.6,
            "max_output_tokens": 150,
        }
        
        # Initialize the model with the correct model name
        model = genai.GenerativeModel(model_name="models/gemini-2.0-flash", 
                                     generation_config=generation_config)
        
        # Generate content using the prompt
        response = model.generate_content(prompt)
        
        # Clean and return the text
        return clean_text(response.text)
        
    except Exception as e:
        # Log the specific error
        logger.error(f"Bot answer error: {str(e)}")
        return f"I'm having trouble processing your request right now. Error: {str(e)}"

def clean_text(text):
    """Clean up the response text"""
    text = text.replace('*', '')
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    text = text.replace('\t', ' ')
    return text.strip()

@app.route('/')
def index():
    return render_template('together.html')

@app.route('/video')
def video():
    return Response(detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/submit', methods=['POST'])
def submit():
    global max_emotion
    logger.info(f"Submit route called with emotion: {max_emotion}")
    response = bot_answer("Who are you?", max_emotion)
    return render_template('together.html', emotion=max_emotion, response=response)

@app.route('/chat', methods=['POST'])
def chat():
    global max_emotion
    user_message = request.json.get('message')
    logger.info(f"Chat request received: {user_message}, Current emotion: {max_emotion}")
    
    bot_response = bot_answer(user_message, max_emotion)
    logger.info(f"Bot response: {bot_response}")
    
    return jsonify({
        'bot_message': bot_response,
        'current_emotion': max_emotion
    })

@app.route('/get_emotion', methods=['GET'])
def get_emotion():
    """API endpoint to get the current detected emotion"""
    global max_emotion
    return jsonify({'emotion': max_emotion}) 

if __name__ == '__main__':
    print("\n🚀 Zaheen is running at http://127.0.0.1:5000")
    print("✨ Emotion detection enabled using DeepFace")
    print("🔄 Starting web server...")
    app.run(debug=True)
