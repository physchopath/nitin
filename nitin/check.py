import os
import cv2
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from keras.models import load_model
import time
from datetime import datetime

app = Flask(__name__)

# Load datasets
academic_df = pd.read_csv('nitin/dataset/academic_records_dataset.csv')
family_df = pd.read_csv('nitin/dataset/family_background_dataset.csv')

# Load QHED Emotion Detection model (assumes you have trained and saved the model)
qhed_model = load_model('preprocessing/my_trained_model_qhed.h5')  # Replace with your model path
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'contempt']

# Emotion to stress level mapping
emotion_stress_map = {
    'happy': 'Low',
    'sad': 'High',
    'contenmt-Angry': 'Low',
    'fear': 'High',
    'disgust': 'Moderate',
    'surprise': 'Low',
    'angry': 'Moderate'
}

# Stress level suggestions

stress_suggestions = {
    'Low': {
        'academic': "Your academic performance seems good. Keep up the good work!",
        'family': "Your family background appears supportive. Maintain healthy relationships.",
        'both': "Both academic and family factors seem positive. Continue your balanced approach.",
        'neither': "You seem to be in a good state. Enjoy your current environment!"
    },
    'Moderate': {
        'academic': "Your academic records show some stress factors. Consider time management techniques.",
        'family': "Your family background shows moderate stress factors. Try open communication.",
        'both': "Both academic and family factors contribute to your stress. Seek balance.",
        'neither': "You seem moderately stressed due to current environment. Take short breaks."
    },
    'High': {
        'academic': "Your academic records indicate high stress. Consider academic counseling.",
        'family': "Your family background shows high stress factors. Family counseling may help.",
        'both': "Both academic and family factors contribute significantly to your stress. Professional help is recommended.",
        'neither': "You seem highly stressed due to current environment. Take a break and seek support."
    }
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect_stress', methods=['POST'])
def detect_stress():
    student_id = request.form['student_id']
    
    # Check if student exists in either dataset
    academic_record = academic_df[academic_df['student_id'] == student_id]
    family_record = family_df[family_df['student_id'] == student_id]
    
    if academic_record.empty and family_record.empty:
        return jsonify({
            'error': 'Student not found in records. Please register as a new student.'
        })
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({'error': 'Could not open video device'})
    
    # Variables for emotion detection
    emotions = []
    start_time = time.time()
    detection_duration = 5  # seconds
    
    while (time.time() - start_time) < detection_duration:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert the frame to grayscale (assuming the model expects grayscale images)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using Haar cascade or any other face detection method
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            face = gray_frame[y:y+h, x:x+w]
            
            # Convert grayscale face to RGB (3-channel) image to match model input
            face_rgb = np.stack([face] * 3, axis=-1)  # Convert to (48, 48, 3)
            
            # Resize the face to the expected input size of the QHED model
            face_rgb = cv2.resize(face_rgb, (48, 48))
            
            # Normalize the image to [0, 1]
            face_rgb = face_rgb.astype('float32') / 255
            face_rgb = np.expand_dims(face_rgb, axis=0)  # Add batch dimension (1, 48, 48, 3)
            
            # Predict emotion using QHED model
            emotion_prob = qhed_model.predict(face_rgb)
            emotion_index = np.argmax(emotion_prob)
            detected_emotion = emotion_labels[emotion_index]
            
            emotions.append(detected_emotion)
    
    cap.release()
    
    if not emotions:
        return jsonify({'error': 'No face detected or emotion recognition failed'})
    
    # Calculate the most frequent emotion
    dominant_emotion = max(set(emotions), key=emotions.count)
    face_stress_level = emotion_stress_map.get(dominant_emotion, 'Moderate')
    
    # Get dataset stress levels if available
    academic_stress = academic_record['stress_level'].values[0] if not academic_record.empty else None
    family_stress = family_record['stress_level'].values[0] if not family_record.empty else None
    
    # Determine stress source
    stress_source = 'neither'
    if academic_stress and family_stress:
        stress_source = 'both'
    elif academic_stress:
        stress_source = 'academic'
    elif family_stress:
        stress_source = 'family'
    
    # Get suggestions
    suggestion = stress_suggestions[face_stress_level][stress_source]
    
    # Prepare response
    response = {
        'student_id': student_id,
        'detected_emotion': dominant_emotion,
        'face_stress_level': face_stress_level,
        'academic_stress': academic_stress,
        'family_stress': family_stress,
        'stress_source': stress_source,
        'suggestion': suggestion,
        'emotion_data': emotions
    }
    
    return jsonify(response)

@app.route('/register_student', methods=['POST'])
def register_student():
    student_id = request.form['new_student_id']
    
    # Check if student already exists
    if student_id in academic_df['student_id'].values or student_id in family_df['student_id'].values:
        return jsonify({'error': 'Student ID already exists'})
    
    # Academic data
    academic_data = {
        'student_id': student_id,
        'gpa': float(request.form['gpa']),
        'attendance_percentage': float(request.form['attendance']),
        'assignment_score': float(request.form['assignment_score']),
        'exam_performance': float(request.form['exam_performance']),
        'project_involvement': float(request.form['project_involvement']),
        'backlogs_count': int(request.form['backlogs_count']),
        'extracurricular_score': float(request.form['extracurricular_score']),
        'academic_warning': int(request.form['academic_warning']),
        'stress_level': request.form['academic_stress_level']
    }
    
    # Family data
    family_data = {
        'student_id': student_id,
        'parental_income': float(request.form['parental_income']),
        'parents_education_level': request.form['parents_education_level'],
        'family_type': request.form['family_type'],
        'no_of_siblings': int(request.form['no_of_siblings']),
        'guardian_type': request.form['guardian_type'],
        'home_study_environment': request.form['home_study_environment'],
        'residential_status': request.form['residential_status'],
        'domestic_issues_reported': int(request.form['domestic_issues_reported']),
        'stress_level': request.form['family_stress_level']
    }
    
    # Add to datasets
    academic_df.loc[len(academic_df)] = academic_data
    family_df.loc[len(family_df)] = family_data
    
    # Save updated datasets
    academic_df.to_csv('nitin/dataset/academic_records_dataset.csv', index=False)
    family_df.to_csv('nitin/dataset/family_background_dataset.csv', index=False)
    
    return jsonify({'success': 'Student registered successfully'})

if __name__ == '__main__':
    app.run(debug=True)
