import sys
import json
import os
from os import path
import cv2

def find_face(imgs):
    result = []

    #Use haar cascade to detect face
    #haar_xml = os.path.join(cv2_dir, 'data/haarcascade_frontalface_default.xml')
    haar_xml = "Model_Files/haarcascade_frontalface_alt.xml"
    if not path.exists(haar_xml):
        print("Cannot find haarcascade_frontalface_alt.xml under Model_Files folder, " + \
              "please download it: https://github.com/opencv/opencv/tree/master/data/haarcascades")
        return
    faceCascade = cv2.CascadeClassifier(haar_xml)

    #Recognize face in each image
    for img_name, img in imgs:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detect_faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.06,
            minNeighbors=5,
        )
        if len(detect_faces) > 0:
            # Store in dict format
            for x,y,w,h in detect_faces:
                #face = {"iname": img_name, "bbox": [float(x),float(y-h*0.31/2),float(w),float(h*1.31)]}
                face = {"iname": img_name, "bbox": [float(x),float(y)*0.8,float(w),float(h*1.2)+float(y)*0.2]}
                result.append(face)
    return result

def main():

    imgs_dir = sys.argv[1]

    #read images and sorted it
    imgs = []
    imgs_dir = imgs_dir + "/images"
    file_list = os.listdir(imgs_dir)
    file_list.sort(key=lambda file_name: int(''.join(filter(str.isdigit, file_name))))

    for img_file in file_list:
        img = cv2.imread(os.path.join(imgs_dir,img_file))
        if img is not None:
            imgs.append([img_file, img])

    result = []
    result = find_face(imgs)

    #dump json
    with open('results.json', 'w') as outfile:
        json.dump(result, outfile)


main()
