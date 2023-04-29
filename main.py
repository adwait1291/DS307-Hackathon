import streamlit as st
import cv2

# Load the trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the camera
cap = cv2.VideoCapture(0)

# Define the camera properties
focal_length = 900
face_width = 14

# Set the distance threshold for face detection
threshold_distance = 30

# Set the title and description of the web app
header = st.empty()

page_bg = f'''
<style>
.center {{
display: flex;
justify-content: center;
align-items: center;
padding-top: 200px;
}}

.top {{
display: flex;
justify-content: center;
align-items: center;
padding-top: 30px;
}}

.stApp {{
background-image: url("https://images.unsplash.com/photo-1554034483-04fda0d3507b?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1770&q=80");
background-size: cover;
}}
</style>
'''

st.markdown(page_bg, unsafe_allow_html=True)

# Define a function to detect faces and return the appropriate text
def detect_faces(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Check if there are any faces detected
    if len(faces) == 0:
        # If no faces are detected within the threshold distance, return "Use for interaction in sign language"
        return 0
    else:
        # If a face is detected within the threshold distance, return "Welcome"
        for (x, y, w, h) in faces:
            distance = focal_length * face_width / w
            if distance <= threshold_distance:
                return 1
            else:
                # If a face is detected outside the threshold distance, return "Use for interaction in sign language"
                return 0

# Define the main function that runs the web app
def main():
    # Define a loop to read the frames from the camera and display the text based on the face detection
    while True:
        # Read the frame from the camera
        ret, frame = cap.read()

        isFace = detect_faces(frame)

        # Detect faces and get the appropriate text
        if isFace == 0:
            header.write("<div class='center'><h1>Use for interaction in sign language!</h1></div>", unsafe_allow_html=True)
        else:
            header.write("<div class='top'><h1>Welcome!</h1></div>", unsafe_allow_html=True)
        # Check for quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

# Call the main function to run the web app
if __name__ == "__main__":
    main()
