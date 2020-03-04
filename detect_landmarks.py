from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import copy

def convert_to_bbox(rect):
	''' 
		take a bounding predicted by dlib and convert it
		to the format (x, y, w, h) as we would normally do
		with OpenCV
	'''
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	return (x, y, w, h)


def convert_to_numpy(shape, dtype="int"):
	'''
		convert facial landmark (x, y)-coordinates to a NumPy array
	'''
	coords = np.zeros((68, 2), dtype=dtype)									# initialize the list of (x, y)-coordinates

	# loop over the 68 facial landmarks
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)						# convert facial landmarks to a 2-tuple of (x, y)-coordinates

	return coords


def main():

    path_to_img = "test.jpg"												# path to image
    model = "shape_predictor_68_face_landmarks.dat"							#path to the face detection model

    detector = dlib.get_frontal_face_detector()								# initialize dlib's face detector
    predictor = dlib.shape_predictor(model)									# create facial landmark predictor

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(path_to_img)
    image = imutils.resize(image, width=500)
    jawline_img = copy.copy(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)												# detect faces in the grayscale image

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)										# determine the facial landmarks for the face region
        shape = convert_to_numpy(shape)										# convert facial landmark (x, y)-coordinates to a NumPy array

        (x, y, w, h) = convert_to_bbox(rect)								# convert dlib's rectangle to a OpenCV-style bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)		# draw the face bounding box
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    	# loop over the (x, y)-coordinates for the facial landmarks
    	# and draw them on the image
        for (x, y) in shape:
            jawline = shape[0:16]
            pts = np.array(jawline, np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(jawline_img, [pts], False, (255,0,0))
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    cv2.imwrite("Output_faces.jpg", image)
    cv2.imwrite("Jawlines.jpg", jawline_img)

if __name__ == '__main__':
    main()
