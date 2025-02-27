import os
import logging
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

models = {
    'brain': load_model('models/brain_tumours_final.h5'),
    'chest': load_model('models/pneumonia_detection_model.h5')
}

thresholds = {
    'brain': 0.5,
    'chest': 0.5
}

def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    logging.debug(f"Preprocessed image shape: {img_array.shape}")
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        cancer_type = request.form['cancer_type']
        uploaded_file = request.files['image']
        if not uploaded_file:
            return "No file uploaded.", 400

        image_path = os.path.join('uploads', uploaded_file.filename)
        os.makedirs('uploads', exist_ok=True)
        uploaded_file.save(image_path)

        model = models.get(cancer_type)
        if not model:
            return "Invalid cancer type selected.", 400

        target_size = (224, 224) if cancer_type == 'brain' else (100, 100)
        image = preprocess_image(image_path, target_size)

        prediction = model.predict(image)
        prediction_value = prediction[0][0]
        logging.info(f"Prediction for {cancer_type}: {prediction_value}")

        threshold = thresholds.get(cancer_type, 0.5)
        result = "Disease Detected" if prediction_value >= threshold else "No Disease Detected"
        logging.info(f"Result: {result}")

        return render_template('predict.html', result=result, value=f"{prediction_value:.4f}", cancer_type=cancer_type)
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return render_template('predict.html', result="Error", value="N/A", cancer_type="N/A"), 500

if __name__ == '__main__':
    app.run(debug=True)
