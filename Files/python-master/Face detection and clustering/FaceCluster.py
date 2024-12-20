import sys
import json
import numpy as np
#import re
import os
from os import path
import cv2
import face_recognition

def crop_face(imgs):
    imgs_crop = []
    #cv2_dir = os.path.dirname(os.path.abspath(cv2.__file__))
    #haar_xml = os.path.join(cv2_dir, 'data/haarcascade_frontalface_alt.xml')

    #Use haar cascade to detect face
    haar_xml = "Model_Files/haarcascade_frontalface_alt.xml"
    if not path.exists(haar_xml):
        print("Cannot find haarcascade_frontalface_alt.xml under Model_Files folder, " + \
              "please download it: https://github.com/opencv/opencv/tree/master/data/haarcascades")
        return
    faceCascade = cv2.CascadeClassifier(haar_xml)

    #Recognize face in each image
    for index in range(0,len(imgs)):
        
        img_name = imgs[index][0]
        img = imgs[index][1]

        #Transfer img to gray img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detect_faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.06,
            minNeighbors=5,
        )
        if len(detect_faces) > 0:
            # Store in dict format
            for x,y,w,h in detect_faces:
                x = int(x)
                w = int(w)
                h = int(h*1.2)+int(y*0.2)
                y = int(y*0.8)
                img_new=img[y:y+h,x:x+w]

                #img_vec = face_recognition.face_encodings(img_new, [(y, x+w, y+h, x)])
                img_vec = face_recognition.face_encodings(img, [(y, x+w, y+h, x)])
                face = {"iname": img_name, "img_vec": img_vec[0], "img":img_new}
                imgs_crop.append(face)


    return imgs_crop

def clusters_list(clus_list):
    dict_list = []
    for index in range(0,len(clus_list)):
        #print("Clust:" + str(index))
        clus_list[index]["img_list"].sort(key=lambda file_name: int(''.join(filter(str.isdigit, file_name))))
        clus_dict = {"cluster_no": index, "element": clus_list[index]["img_list"]}
        dict_list.append(clus_dict)
    return dict_list

def find_min_pairs(sl_clusters):

    pairs = []
    min_dis = 0
    size_clus = len(sl_clusters)

    # Get min dis pair (Euclidean distance)
    euc_matrix = np.zeros((size_clus,size_clus)) - np.ones((size_clus,size_clus))

    for c1_index in range(0,size_clus-1):
        for c2_index in range(c1_index+1,size_clus):
            euc_matrix[c1_index,c2_index] =  np.linalg.norm(sl_clusters[c1_index]["vec"]-sl_clusters[c2_index]["vec"])

    #print(np.min(euc_matrix[np.where(euc_matrix>-1)]))

    # Min dis pairs
    min_pair = np.where( euc_matrix == np.min(euc_matrix[np.where(euc_matrix>-1)]))

    return [min_pair[0][0],min_pair[1][0]]

def face_cluster(imgs_crop, cluster_k):
    clusters = []

    #single link clusters
    sl_clusters = []
    for index in range(0,len(imgs_crop)):
        cluster_info = {"img_list":[imgs_crop[index]["iname"]],"vec":imgs_crop[index]["img_vec"]}
        sl_clusters.append(cluster_info)

    while(1):
        size_clus = len(sl_clusters)

        # Get min dis pair
        pairs = find_min_pairs(sl_clusters)
        cluster_info = {"img_list": sl_clusters[pairs[0]]["img_list"] + sl_clusters[pairs[1]]["img_list"],
                        "vec": (sl_clusters[pairs[0]]["vec"] + sl_clusters[pairs[1]]["vec"])/2 }
        sl_clusters.append(cluster_info)
        sl_clusters.pop(pairs[0])
        sl_clusters.pop(pairs[1]-1)

        # Cluster size == k, stop slc
        if len(sl_clusters) < cluster_k+1:
            break

    # Generate json format info
    clusters = clusters_list(sl_clusters)

    return clusters
def dump_cluster_img(imgs_crop, clusters):

    for clu_index in range(0,len(clusters)):
        clu_image = np.zeros((600,600,3), np.uint8)
        clu_x = 0
        clu_y = 0

        for ele_index in range(0, len(clusters[clu_index]["element"])):
            img_index = int(''.join(filter(str.isdigit, clusters[clu_index]["element"][ele_index])))-1

            # Crop face to square
            crop_img = imgs_crop[img_index]["img"]
            y,x,r = crop_img.shape
            if y>x:
                new_len = int((y-x)/2)
                crop_img = crop_img[new_len:y-new_len,:]
            elif y<x:
                new_len = int((x-y)/2)
                crop_img = crop_img[:,new_len:x-new_len]
            crop_img = cv2.resize(crop_img, (200,200), interpolation = cv2.INTER_AREA)

            # Paste together
            clu_image[clu_y:clu_y+200, clu_x:clu_x+200] = crop_img

            if clu_x == 400:
                clu_x = 0
                clu_y += 200
            else:
                clu_x += 200
        cv2.imwrite("cluster"+ str(clu_index) + ".jpg", clu_image)

def main():

    imgs_dir = sys.argv[1]

    #read images
    imgs = []
    cluster_k = 0
    cluster_k = int(''.join(filter(str.isdigit, imgs_dir)))

    file_list = os.listdir(imgs_dir)
    file_list.sort(key=lambda file_name: int(''.join(filter(str.isdigit, file_name))))

    for img_file in file_list:
        img = cv2.imread(os.path.join(imgs_dir,img_file))
        if img is not None:
            imgs.append([img_file, img])
    # Step1 and 2
    imgs_crop = []
    imgs_crop = crop_face(imgs)

    # Step 3
    clusters = []
    clusters = face_cluster(imgs_crop, cluster_k)

    #dump json
    with open('clusters.json', 'w') as outfile:
        json.dump(clusters, outfile)

    #Dump cluster img for report
    #dump_cluster_img(imgs_crop, clusters)

main()
