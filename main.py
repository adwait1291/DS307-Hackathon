import streamlit as st
import cv2
import time
import numpy as np

st.set_page_config(layout="wide")

# Load the trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the camera
cap = cv2.VideoCapture(0)




import cv2
import base64
import numpy as np
import requests
import time
import json

# Construct the Roboflow Infer URL
# (if running locally replace https://classify.roboflow.com/ with eg http://127.0.0.1:9001/)
upload_url = "https://classify.roboflow.com/abcd-ifyky/1?api_key=NkdlEq0xkYZXpHT00Yk3"





# Set the width and height of the camera feed to 640x480
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 790)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# Define the camera properties
focal_length = 900
face_width = 14

# Set the distance threshold for face detection
threshold_distance = 30

from roboflow import Roboflow
rf = Roboflow(api_key="NkdlEq0xkYZXpHT00Yk3")
project = rf.workspace().project("abcd-ifyky")
model = project.version(1).model


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
            

def get_label(img):
    # response = model.predict(image).json()
    # label = response["predictions"][0]["class"] if response["predictions"] else ""
    # return label

    # Encode image to base64 string
    retval, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)

    # Get prediction from Roboflow Infer API
    resp = requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    }, stream=True)

    preds = resp.json()
    return preds["predictions"][0]["class"]


# Define the main function that runs the web app
def main():
    
    # Define a loop to read the frames from the camera and display the text based on the face detection
    header = st.empty()

    col1, col2 = st.columns([2, 1])


    with col2:
        header2 = st.empty()

    with col1:
        timer_text = st.empty()
        placeholder = st.empty()

    st.session_state.timer = 5

    last_detection_time = time.time() - 30

    generated_text = ""

    while True:
        # Read the frame from the camera
        _, frame = cap.read()


        # Detect faces and get the appropriate text
        
        with col1:
            isFace = detect_faces(frame)
            if isFace == 0 and time.time() - last_detection_time >= 30:
                header.write("<div class='center'><h1>Use for interaction in sign language!</h1></div>", unsafe_allow_html=True)
                generated_text = ""
        
            else:
                header.write("<div class='top'><h1>Welcome!</h1></div>", unsafe_allow_html=True)
                frame = cv2.resize(frame, (0, 0), fx=0.45, fy=0.5)

                # Convert the frame from BGR (OpenCV default) to RGB (Streamlit default)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                placeholder.image(frame, channels="RGB")

                last_detection_time = time.time()

                if "start_time" not in st.session_state:
                    st.session_state.start_time = time.time()
                elif time.time() - st.session_state.start_time > 5:
                    txt = get_label(frame)
                    generated_text+=txt
                    header2.write(f"<h1>{generated_text}</h1>", unsafe_allow_html=True)
                    st.session_state.start_time = time.time()

                # Display the timer in seconds
                remaining_time = int(5 - (time.time() - st.session_state.start_time))
                timer_text.text(f"Timer: {remaining_time}s")


                # get_label(frame)


        # Check for quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

# Call the main function to run the web app
if __name__ == "__main__":
    main()
