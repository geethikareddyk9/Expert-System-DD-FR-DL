import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

app = Flask(__name__)

from keras.models import Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.applications import DenseNet201, DenseNet121

from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D

# Define quantum ReLU
def q_relu(x):
    """Quantum ReLU activation function."""
    return tf.where(tf.greater(x, 0), x, 0.01 * x)

# define Model
def build_densenet():
    densenet = DenseNet201(weights='imagenet', include_top=False)

    input = Input(shape=(224,224,3))
    x = Conv2D(3, (3, 3), padding='same')(input)

    x = densenet(x)

    # Add a global average pooling layer
    x = GlobalAveragePooling2D()(x)

    # Add dense layers with 1024, 512, and 128 units and ReLU activation
    x = Dense(1024, activation= q_relu)(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation= q_relu)(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation= q_relu)(x)
    x = Dropout(0.2)(x)

    # Multi-output layer
    output = Dense(10, activation='softmax', name='root')(x)

    # Create the model
    model = Model(input, output)

    return model

# Build the model
model = build_densenet()
model.summary()

# Load the weights
model.load_weights('best_model_9958.weights.h5')

# # Load the trained model
# model = load_model('best_model_9923.keras')

# Define the list of disease classes
disease_class = ['Tomato_Early_blight', 'Tomato_Spider_mites_Two_spotted_spider_mite',
                 'Tomato_Target_Spot', 'Tomato_YellowLeaf_Curl_Virus',
                 'Tomato_mosaic_virus', 'Tomato_Bacterial_spot',
                 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                 'Tomato_healthy']

# Fertilizer recommendation dictionary
fertilizer_recommendation = {
    'Tomato_Early_blight': 'Use a balanced fertilizer with a ratio of 10-10-10.',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Use a fertilizer with a high nitrogen content.',
    'Tomato__Target_Spot': 'Use a fertilizer with a high phosphorus content.',
    'Tomato_YellowLeaf_Curl_Virus': 'Use a fertilizer with a high potassium content.',
    'Tomato_mosaic_virus': 'Use a fertilizer with a high calcium content.',
    'Tomato_Bacterial_spot': 'Use a fertilizer with a high copper content.',
    'Tomato_Late_blight': 'Use a fertilizer with a high manganese content.',
    'Tomato_Leaf_Mold': 'Use a fertilizer with a high sulfur content.',
    'Tomato_Septoria_leaf_spot': 'Use a fertilizer with a high zinc content.',
    'Tomato_healthy': 'No fertilizer recommendation needed.'
}

root_cause = {
    'Tomato_Early_blight': 'Fungal infection caused by Alternaria solani fungus',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Infestation by Tetranychus urticae, commonly known as two-spotted spider mites',
    'Tomato__Target_Spot': 'Fungal infection caused by Corynespora cassiicola',
    'Tomato_YellowLeaf_Curl_Virus': 'Viral infection caused by Tomato yellow leaf curl virus (TYLCV)',
    'Tomato_mosaic_virus': 'Viral infection caused by Tomato mosaic virus (ToMV)',  
    # 'Tomato_Bacterial_spot': 'Bacterial infection caused by Xanthomonas campestris bacteria',
    'Tomato_Bacterial_spot': 'Bacterial infection caused by Xanthomonas perforans or Xanthomonas gardneri or Xanthomonas campestris bacteria',
    'Tomato_Leaf_Mold': 'Fungal infection caused by Passalora fulva (formerly Fulvia fulva)',
    'Tomato_Late_blight': 'Fungal infection caused by Phytophthora infestans',
    'Tomato_Septoria_leaf_spot': 'Fungal infection caused by Septoria lycopersici',
    'Tomato_healthy': 'Healthy tomato plants without any visible diseases or pests',
}

def preprocess_image(image_path):
    image = tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=(224, 224)
    )
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message='No selected file')
    if file:
        # Create directory if it doesn't exist
        save_dir = 'images'
        os.makedirs(save_dir, exist_ok=True)
        
        # Save file
        file_path = os.path.join(save_dir, 'leaf_image.jpg')
        file.save(file_path)
        
        image = preprocess_image(file_path)
        prediction = model.predict(np.expand_dims(image, axis=0))
        predicted_class_index = np.argmax(prediction)
        predicted_class = disease_class[predicted_class_index]
        predicted_root_cause = root_cause[predicted_class]
        fertilizer_recommendation_predicted = fertilizer_recommendation[predicted_class]
        
        # # Get the predicted disease class
        # predicted_disease = disease_class[predicted_class_index]
        
        # # Initialize variables for root cause and fertilizer recommendation
        # predicted_root_cause = "Root cause information not available"
        # fertilizer_recommendation_predicted = "Fertilizer recommendation not available"
        
        # # Check if the predicted disease is in the disease_info dictionary
        # if predicted_disease in disease_class:
        #     # Get disease information
        #     predicted_class = disease_info[predicted_disease]
        #     predicted_root_cause = predicted_class.get('root_cause', root_cause)
        #     fertilizer_recommendation_predicted = predicted_class.get('fertilizer_recommendation', fertilizer_recommendation)

        return render_template('index.html', image_path=file_path, prediction=predicted_class, root_cause=predicted_root_cause, fertilizer_recommendation=fertilizer_recommendation_predicted)

if __name__ == '__main__':
    app.run(debug=True)