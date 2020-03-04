# Facial-Landmark_detection### How to run

## How to setup

Fire the following commands in the terminal :

1. git clone https://github.com/PulkitMishra/Facial-Landmark-Detection
2. virtualenv -p python3 hike_env
3. source hike_env/bin/activate
4. cd Facial-Landmark-Detection
5. pip3 install -r requirements.txt
6. Download the dlib’s pre-trained facial landmark detector model from [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and save it in the root of this repository.

## How to run

1. Change the value of `path_to_img` in the detect_landmarks.py file
2. Execute the following command

```
  python detect_landmarks.py
```

## Output

1. 68 face landmarks for each face in the image is saved as Output_faces.jpg in the root of the directory.
2. The curve for jawlines is saved as Jawlines.jpg in the root of the directory.
