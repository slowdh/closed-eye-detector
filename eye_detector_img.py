import cv2
import dlib

# Load the detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# read the image
img = cv2.imread('test_closed.png')

# Define 'Closed' eye with EAR
# p1 , ..., p6 is (1,1) shape tuple consists of x, y coordinate value of an eye.
# eye parameter is a list consist of 6 different points
def get_ear_ratio(eye):
    def _get_euclidean_distance_2d(p0, p1):
        x_p0 = p0[0]
        y_p0 = p0[1]
        x_p1 = p1[0]
        y_p1 = p1[1]
        distance = ((x_p0 - x_p1) ** 2 + (y_p0 - y_p1) ** 2) ** 0.5
        return distance

    len_horizontal = _get_euclidean_distance_2d(eye[0], eye[3])
    len_vertical_1 = _get_euclidean_distance_2d(eye[1], eye[5])
    len_vertical_2 = _get_euclidean_distance_2d(eye[2], eye[4])
    ear_ratio = (len_vertical_1 + len_vertical_2) / (2 * len_horizontal)
    return ear_ratio

# Convert image into grayscale
gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

# Use detector to find landmarks
faces = detector(gray)
for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()

    # Create landmark object
    landmarks = predictor(image=gray, box=face)

    # Loop through all the points
    eye_left_lst = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
    eye_right_lst = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
    for point in eye_left_lst + eye_right_lst:
        cv2.circle(img=img, center=(point[0], point[1]), radius=3, color=(0, 255, 0), thickness=-1)

    # Detect closed eye
    ear_ratio_of_left_eye = get_ear_ratio(eye_left_lst)
    ear_ratio_of_right_eye = get_ear_ratio(eye_right_lst)
    threshold = 0.2

    if ear_ratio_of_left_eye < threshold and ear_ratio_of_right_eye < threshold:
        cv2.putText(img, 'Closed', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        print('Closed')
    else:
        cv2.putText(img, 'Open', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        print('Open')

# Show the image
cv2.imshow(winname="Closed eye detector", mat=img)

# Exit when escape is pressed
cv2.waitKey(delay=0)
cv2.destroyAllWindows()