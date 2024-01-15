from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import cv2

# Load the Keras pretrained model
model = load_model('model.h5')

app = Flask(__name__)

@app.route('/recognize', methods=['POST'])
def recognize():
    print("Recognize")

    # Get the input image from the JSON
    data = request.get_json()
    image = data.get('image')

    # Convert the list to a numpy array
    image = np.array(image)

    # Resize the image to 28x28, and adapt it to what the model expects
    image = np.reshape(image, (-1, 28, 28, 1))

    # Perform prediction using the loaded model
    predictions = model.predict(image)

    # Save the image to disk for testing purposes
    cv2.imwrite("predicted.png", image[0] * 255)

    # Get the index of the maximum value from the prediction
    max_value = predictions[0].argmax(axis=0)

    print("Returning...", max_value)

    # Return the prediction as a JSON
    return jsonify({'result': int(max_value), 'predictions': predictions[0].tolist()})

@app.route('/convert', methods=['POST'])
def convert_image():
    # Process the image here
    data = request.get_json()
    image_path = data.get('path')

    # Load the image using cv2
    image = cv2.imread(image_path)

    # Resize the image to 28x28
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

    # Convert the image to black and white
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize the image
    image = image / 255.0

    # Reshape the image to match the model's input shape
    image = np.reshape(image, (1, 28, 28, 1))

    # Convert the image to a string representation
    image_string = np.array2string(image)

    # Return the string representation of the image
    return image.tolist()

if __name__ == '__main__':
    app.run(debug=True)