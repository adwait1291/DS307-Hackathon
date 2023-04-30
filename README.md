# ASL Assist

ASL Assist is a project that aims to assist the deaf community in booking tickets for the Bus Rapid Transit System (BRTS) in a more efficient manner. The system captures character by character footage of people doing sign language and converts it into text. The text is then converted into speech that can be heard by the counter person, thus enabling a smoother and more efficient ticket booking process.

## Installation

To install ASL Assist, you will need to have Python 3.7 or higher installed on your system. Once you have Python installed, follow these steps:

1. Clone the ASL Assist repository to your local machine.
2. Install the required packages using pip: `pip install -r requirements.txt`
3. Run the `main.py` file: `streamlit run app.py`

## Usage

To use ASL Assist, follow these steps:

1. Open the ASL Assist web application by running `streamlit run app.py`.
2. Bring face within 50cm proximity to begin capturing sign language footage.
3. Sign the name of the station you wish to book a ticket for character by character
4. Show the "Space" symbol to end the footage capture process.
5. The captured footage will be processed by the system and the corresponding text will be displayed on the screen.
6. "Please book a ticket for <station_name>" will be automatically spoken out loud in the counter person speaker.
7. Proceed to the counter and collect your ticket.

## Technologies Used

ASL Assist utilizes the following technologies:

- [Roboflow](https://roboflow.com/) for object detection and image processing
- [Streamlit](https://streamlit.io/) for building the web application
- [gTTS](https://pypi.org/project/gTTS/) for text-to-speech conversion
- [OpenCV](https://opencv.org/) for video processing and analysis

## Credits

ASL Assist was developed by Adwait Kesharwani and Yashashvi Singh as part of DS307: Innovation and Entrpeneurship Seminal Hackathon project. It was inspired by the need to make ticket booking easier for the deaf community. Special thanks to Dr.Dashrath R Shetty and Dr.Manjunath KV for their guidance and support.
