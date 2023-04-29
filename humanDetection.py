# import cv2

# # Load the trained face detection model
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Initialize the camera
# cap = cv2.VideoCapture(0)

# # Define the camera properties
# focal_length = 900
# face_width = 14

# # Loop through the frames from the camera
# while True:
#     # Read the frame
#     ret, frame = cap.read()

#     if not ret:
#         continue

#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # Detect faces in the frame
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
#     # If one face is detected, draw a bounding box around it
#     if len(faces) == 1:
#         x, y, w, h = faces[0]
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
#         # Calculate the distance to the face
#         distance = focal_length * face_width / w
        
#         # Display a message based on the distance
#         if distance < 20:
#             cv2.putText(frame, "Too close! Distance: {:.2f} cm".format(distance), (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#         else:
#             cv2.putText(frame, "Distance: {:.2f} cm".format(distance), (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Show the frame
#     cv2.imshow('frame',frame)
    
#     # Check for quit command
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the camera and close the window
# cap.release()
# cv2.destroyAllWindows()
