Face Recognition and Classification System

This repository hosts a Flask-based web application that performs real-time face recognition and classification using a pre-trained PyTorch model. Users can dynamically update the model with labeled images, select labels for live video detection, and identify faces in video streams.

![Screenshot 2024-11-28 105634](https://github.com/user-attachments/assets/42638007-6163-4349-a182-19efe25fdf1d)
![Screenshot 2024-11-28 104833](https://github.com/user-attachments/assets/282b0f70-f46a-4933-867c-cc8e6268aec4)



Features

Real-Time Face Detection: Employs OpenCV's Haar Cascade for detecting faces in live video streams.

Dynamic Model Retraining: Allows users to upload labeled images and retrain the model on-the-fly.

Interactive UI: Provides an intuitive interface to upload images, select labels, and view live detection results.

Pre-Trained PyTorch Model: Utilizes a pre-trained model with customizable layers for classification.

Requirements

Ensure the following dependencies are installed:

Python 3.7+

Flask

OpenCV

PyTorch

torchvision

PIL (Pillow)

NumPy

Setup Instructions

1. Clone the Repository

2. Install Dependencies

3. Prepare the Environment

Place the pre-trained model file (facenet_class_traced.pt) in the project directory.

Ensure a folder structure exists for uploading labeled images (created automatically during runtime).

4. Run the Application

$ python app.py

Access the application at http://127.0.0.1:5000/.

Application Workflow

Homepage

View and select available labels.

Upload labeled images to dynamically update the model.

Upload Labeled Images

Upload images associated with specific labels for model retraining.

Automatically organizes uploaded images into labeled folders.

Live Video Detection

Streams video from a connected camera.

Detects faces in real-time and highlights matches with the selected label.

Retraining Process

Uploading labeled images triggers the following steps:

Dataset Preparation: Organizes images into labeled datasets.

Model Training: Fine-tunes the classifier layer of the pre-trained model.

Model Saving: Saves the updated model and reloads it for inference.

File Structure

    face-recognition-system/
    ├── app.py                # Main Flask application
    ├── facenet_class_traced.pt # Pre-trained model file
    ├── uploads/             # Folder for uploaded labeled images
    ├── templates/          # HTML templates for the Flask app
    └── static/             # Static assets (CSS, JS)

Limitations

Retraining Duration: Time required depends on the size of the dataset.

Detection Accuracy: May vary with image quality, lighting conditions, and the dataset.

Future Enhancements

Add GPU support for faster training and inference.

Improve pre-processing techniques for enhanced detection accuracy.

Enable multi-label detection in live video streams.

Integrate additional face recognition models.


OpenCV for its efficient computer vision tools.

Flask for a lightweight and flexible web framework.

Contributions

Contributions are welcome! Please submit issues or pull requests to enhance this repository.

