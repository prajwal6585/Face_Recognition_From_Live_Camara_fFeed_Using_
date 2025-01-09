
from flask import Flask, Response, render_template, request, jsonify
import cv2
import numpy as np
import torch
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
import os
from PIL import Image
import torch.nn as nn
import torch.optim as optim

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the traced PyTorch model
model_path = 'facenet_class_traced.pt'

model = torch.jit.load(model_path, map_location=torch.device('cpu'))
model.eval()  # Set the model to evaluation mode


# Define class labels (this will be updated dynamically)
class_labels = ['Athul', 'Mohith', 'Prajna', 'Prajwal', 'prasid', 'Rithesh', 'Puneeth', 'ritheshr']

# Define transformation for input images
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load OpenCV's Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize camera
camera = cv2.VideoCapture(0)

# Placeholder for selected label
selected_label = None


def retrain_model():
    """
    Retrain the model with the newly uploaded data.
    """
    global model, class_labels

    # Load the dataset from the upload directory
    dataset = datasets.ImageFolder(root=app.config['UPLOAD_FOLDER'], transform=transform)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Update class labels from the dataset
    class_labels = dataset.classes

    # Define a simple classifier model (modify based on your existing model)
    num_classes = len(class_labels)
    classifier = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )

    # Attach the classifier to the existing model
    base_model = models.resnet18(pretrained=True)
    base_model.fc = classifier

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(base_model.parameters(), lr=0.001)

    # Fine-tune the model
    base_model.train()
    for epoch in range(5):  # Short retraining for simplicity
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = base_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Save the retrained model (optional)
    torch.jit.save(torch.jit.script(base_model), model_path)

    # Load the retrained model for inference
    model = torch.jit.load(model_path)
    model.eval()


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', labels=class_labels)


@app.route('/upload', methods=['POST'])
def upload_image():
    """
    Upload an image, associate it with a label, and retrain the model.
    """
    global class_labels
    file = request.files['image']
    label = request.form['label']

    if file and label:
        # Save the uploaded image
        label_folder = os.path.join(app.config['UPLOAD_FOLDER'], label)
        os.makedirs(label_folder, exist_ok=True)
        filepath = os.path.join(label_folder, file.filename)
        file.save(filepath)

        # Retrain the model with the new data
        retrain_model()

        return jsonify({'message': f'Image uploaded for label: {label}', 'labels': class_labels})

    return jsonify({'error': 'Failed to upload image or label missing'})


@app.route('/select_label', methods=['POST'])
def select_label():
    """
    Handle the label selection for live video detection.
    """
    global selected_label
    selected_label = request.form['label']
    return jsonify({'message': f'Selected label: {selected_label}'})


def process_frame():
    """
    Continuously captures video frames, detects faces, and checks for the selected label.
    """
    global selected_label

    while True:
        ret, frame = camera.read()
        if not ret or not selected_label:
            continue

        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the face region
            face = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert to RGB

            try:
                # Preprocess the face for the model
                face_tensor = transform(Image.fromarray(face_rgb))
                face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension

                # Perform inference with the model
                with torch.no_grad():
                    output = model(face_tensor)
                    predicted_index = torch.argmax(output, dim=1).item()
                    predicted_label = class_labels[predicted_index]

                # Check if the predicted label matches the selected label
                if predicted_label == selected_label:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Found: {predicted_label}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error processing face: {e}")
                continue

        # Encode the frame as JPEG and yield it
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(process_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
