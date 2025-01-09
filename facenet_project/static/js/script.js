const videoElement = document.getElementById('video-feed');
const resultElement = document.getElementById('result');

// Start the video stream from the webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        videoElement.srcObject = stream;
    })
    .catch(err => {
        console.error("Error accessing the camera: ", err);
    });

// Handle the image upload form submission
document.getElementById('upload-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const formData = new FormData();
    const imageFile = document.getElementById('image-upload').files[0];
    formData.append('image', imageFile);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.prediction) {
            resultElement.innerText = 'Target found: ' + data.prediction;
        } else {
            resultElement.innerText = 'No target detected';
        }
    })
    .catch(error => {
        console.error('Error uploading image:', error);
    });
});

// Optional: Function to handle face detection in real-time video
function detectFaceInVideo() {
    // Use OpenCV.js or other face detection methods here
    // Example: If face detected, draw rectangle and alert

    // For example, you can use OpenCV.js for face detection on the video feed.
}

setInterval(detectFaceInVideo, 100);  // Check for faces every 100ms
