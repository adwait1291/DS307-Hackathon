import streamlit as st
import cv2
import time
import numpy as np

st.set_page_config(layout="wide")

# Load the trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the camera
cap = cv2.VideoCapture(0)


stations_dict = {
"dharwadbrtsterminal":"Dharwad BRTS Terminal",
"jubileecircle":"Jubilee Circle","courtcircle":"Court Circle",
"nttf":"NTTF","hosayallapurcross":"Hosa Yallapur Cross","tollnakka":"Toll Nakka","vidyagiri":"Vidyagiri",
"gandhinagar":"Gandhi Nagar","lakamanahalli":"Lakamanahalli","sattur":"Sattur","sdmmedicalcollege":"SDM Medical College","navalurrailwaystation":"Navalur Railway Station",
"kmf":"KMF-1","rayapur":"Rayapur","iskcontemple":"Iskcon Temple","rtooffice":"RTO Office","navanagar":"Navanagar","ampcgate":"AMPC 3rd Gate","shantiniketan":"Shantiniketan",
"baridevarakoppa":"Baridevarakoppa","unakallake":"Unakal Lake","unakal":"Unakal","unakalcross":"Unakal Cross","bvbcollege":"BVB College","vidyanagar":"Vidyanagar","kims":"KIMS",
"hosurregionalterminal":"Hosur Regional Terminal","hosurcross":"Hosur Cross","drbrambedkarcircle":"Dr. B R Ambedkar Circle","huballicentralbusterminal":"Huballi Central Bus Terminal",
"cbthuballi":"CBT Huballi"
 }

stations = list(stations_dict.keys())
print(stations)

from gtts import gTTS
import os

# Define the text to convert to speech
def play_sound(text):
    # Create a gTTS object and generate the audio file
    text = "Please book a ticket for "+text
    tts = gTTS(text=text, lang='en')
    tts.save("test.mp3")
    # Play the audio file using the default media player
    os.system("afplay test.mp3")



import cv2
import base64
import numpy as np
import requests
import time
import json

# Construct the Roboflow Infer URL
# (if running locally replace https://classify.roboflow.com/ with eg http://127.0.0.1:9001/)
# upload_url = "https://classify.roboflow.com/text-elnqt/1?api_key=tXBF80SyixvY9Se6iorC"

upload_url = "https://classify.roboflow.com/text-elnqt/1?api_key=tXBF80SyixvY9Se6iorC"

from roboflow import Roboflow
rf = Roboflow(api_key="tXBF80SyixvY9Se6iorC")
project = rf.workspace().project("sign-detection-6ibui")
model = project.version(1).model

# infer on a local image


# Set the width and height of the camera feed to 640x480
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 790)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# Define the camera properties
focal_length = 900
face_width = 14

# Set the distance threshold for face detection
threshold_distance = 30


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
    # response = model.predict(img, confidence=20, overlap=30).json()
    # label = response["predictions"][0]["class"] if response["predictions"] else ""
    # return label



# image classification
    # Encode image to base64 string
    retval, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)

    # Get prediction from Roboflow Infer API
    resp = requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    }, stream=True)

    preds = resp.json()
    return preds["predictions"][0]["class"] if preds["predictions"] else ""


# Define the main function that runs the web app
def main():
    
    # Define a loop to read the frames from the camera and display the text based on the face detection
    header = st.empty()

    col1, col2 = st.columns([2, 1])


    with col2:
        header2 = st.empty()
        station_list = st.empty()

    with col1:
        timer_text = st.empty()
        placeholder = st.empty()

    st.session_state.timer = 5

    last_detection_time = time.time() - 30

    generated_text = ""



    x, y, w, h = int(cap.get(3)/2), 0, int(cap.get(3)/2), int(cap.get(4))


    while True:
        # Read the frame from the camera
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        bimg = frame[y:y+h, x:x+w]
        # cv2.imshow("Cropped ROI", bimg)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Detect faces and get the appropriate text
        
        with col1:
            isFace = detect_faces(frame)
            if isFace == 0 and time.time() - last_detection_time >= 30:
                header.write("<div class='center'><h1>Use Me for Interaction in Sign Language!</h1></div>", unsafe_allow_html=True)
                generated_text = ""
        
            else:
                header.write("<div class='top'><h1>Please Sign the Name of Your Station Here.</h1></div>", unsafe_allow_html=True)


                frame = cv2.resize(frame, (0, 0), fx=0.45, fy=0.5)
                

                # Convert the frame from BGR (OpenCV default) to RGB (Streamlit default)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                bimg = cv2.cvtColor(bimg, cv2.COLOR_BGR2RGB)


                placeholder.image(frame, channels="RGB")

                last_detection_time = time.time()

                if "start_time" not in st.session_state:
                    st.session_state.start_time = time.time()
                    if not generated_text:
                        station_list.write("\n".join([f"- {stations_dict[item]}" for item in stations]), allow_markdown=True)

                elif time.time() - st.session_state.start_time > 5:
                    txt = get_label(bimg)
                    if txt!="nothing" and txt!="space" and txt!="del":
                        generated_text+=txt

                    if txt=="del":
                        generated_text = generated_text[:-1]
                    matches = [word for word in stations if word.startswith(generated_text.lower())]
                    

                    station_list.write("\n".join([f"- {stations_dict[item]}" for item in matches]), allow_markdown=True)

                    header2.write(f"<h1>{generated_text}</h1>", unsafe_allow_html=True)
                    st.session_state.start_time = time.time()

                    if len(matches)==1:
                        play_sound(matches[0])


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
