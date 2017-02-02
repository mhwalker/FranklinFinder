from subprocess import call
import matplotlib as plt
import cv2
import csv
from keras.models import load_model
from keras import backend as K
import numpy as np
from scipy.spatial import KDTree
import pickle
import os.path
import json

model = load_model("dog_cnn_inception.1.10.keras")
picklename = "petfinderPurebred_dog_cnn_inception.pickle"
with open(picklename, 'rb') as handle:
    b = pickle.load(handle)
farray = np.array(b.values())
tree = KDTree(farray)
photo_ids = b.keys()

with open("small_dog_info.json","rb") as dog_file:
    dog_info = json.load(dog_file)

def processImage(filepath,imagefilename):
    shrunkImageName, scaleRatio = shrinkImage(filepath,imagefilename)
    clippedImageName, boundingBox, imageScore, isCached = runDarknet(filepath,shrunkImageName)
    print clippedImageName, boundingBox, imageScore
    if imageScore == 0: return clippedImageName, [], boundingBox
    #boundedImageName = drawBoundingBoxOnImage(filepath,imagefilename,boundingBox,scaleRatio,isCached)
    boundedImageName = drawBoundingBoxOnImage(filepath,shrunkImageName,boundingBox,1.0,isCached)
    features = getFeatures(filepath,clippedImageName,isCached)
    results = getNeighbors(features)
    dogData = getDogData(results)
    return boundedImageName,results,dogData

def shrinkImage(filepath,imagefilename):

    #set our target height
    targetHeight = 600.0

    #read in original image
    imagename = filepath+"/"+imagefilename
    img = cv2.imread(imagename)

    scaleFactor = img.shape[0]/targetHeight

    #set output image name
    filebase = imagefilename.rsplit('.', 1)[0]
    fileext = imagefilename.rsplit('.', 1)[1]

    newimagefilename = filebase+"_shrunk."+fileext

    #check that the image is larger than our target size, if not return
    if img.shape[0] <= targetHeight:
        cv2.imwrite(filepath+"/"+newimagefilename,img)
        return newimagefilename, 1.0

    #calculate aspect ratio and target width to maintain image dimensions when shrinking
    aspectRatio = float(img.shape[0])/float(img.shape[1])
    targetWidth = targetHeight / aspectRatio

    #resize image
    imgout = cv2.resize(img, (int(targetWidth), int(targetHeight)), cv2.INTER_LINEAR)

    #write the image to same directory, but with an extra tag
    cv2.imwrite(filepath+"/"+newimagefilename,imgout)

    #return new filename and ratio for showing bounding box
    return newimagefilename, scaleFactor

def runDarknet(filepath,imagefilename):
    #base command to run darknet
    command=["darknet","detect","./cfg/yolo.cfg","./yolo.weights"]
    #add file to search
    command.append(filepath+"/"+imagefilename)
    #add output prefices
    command.append("-prefix")
    filebase=imagefilename.rsplit('.', 1)[0]
    fileext=imagefilename.rsplit('.', 1)[1]
    ofname = filebase+"_clipped."+fileext
    prefix = filepath+"/"+filebase+"_result"

    #check if we have processed this file already
    isCached = False
    if os.path.isfile(prefix+".csv"):
        isCached = True

    command.append(prefix)

    #run darknet
    if not isCached:
        call(command)

    #read darknet output and find "best" dog
    mImg = None
    mImgScore = 0
    mBox = (0,0,0,0)
    ndScore = 0
    ndLabel = ""
    with open(prefix+".csv", 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == "image_id":continue
            score = float(row[-2])
            if score > ndScore:
                ndScore = score
                ndLabel = row[-1]
            #only look at dogs that were detected
            if row[-1] != "dog":continue
            #choose the "best" probability dog
            if score < mImgScore: continue
            mImgScore = score
            
            #select the dog box coordinates
            xmin=int(row[4])
            xmax=int(row[5])
            ymin=int(row[6])
            ymax=int(row[7])

            #read the image in
            imagename=row[0]
            img = cv2.imread(imagename)
            #clip the image
            roi = img[ymin:ymax,xmin:xmax]
            mImg = roi
            #save bounding box: (x,y,w,h)
            mBox = (xmin,ymin,xmax-xmin,ymax-ymin)
    #save the clipped dog
    if not isCached:
        cv2.imwrite(filepath+"/"+ofname,mImg)
    #return bounding box coordinates and new filename
    if mImgScore == 0: return ndLabel,filebase+"_result.png",mImgScore,isCached
    return ofname,mBox,mImgScore,isCached

def drawBoundingBoxOnImage(filepath,imagename,boundingBox,scaleFactor,isCached):
    filebase=imagename.rsplit('.', 1)[0]
    fileext=imagename.rsplit('.', 1)[1]

    outfilename = filebase+"_bounding."+fileext

    if isCached: return outfilename

    img = cv2.imread(filepath+"/"+imagename)
    x1 = int(boundingBox[0] * scaleFactor)
    y1 = int(boundingBox[1] * scaleFactor)
    x2 = int((boundingBox[0] + boundingBox[2])*scaleFactor)
    y2 = int((boundingBox[1] + boundingBox[3])*scaleFactor)
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),5)

    cv2.imwrite(filepath+outfilename,img)

    return outfilename

def getFeatures(filepath,imagename,isCached):
    filebase=imagename.rsplit('.', 1)[0]
    fileext=imagename.rsplit('.', 1)[1]
    outfilename = filepath+filebase+".npy"

    if isCached:
        result = np.load(outfilename)
        return result

    img = cv2.imread(filepath+"/"+imagename)
    img = cv2.resize(img, (299,299), cv2.INTER_LINEAR)
    X_test = np.array([img])
    print X_test.shape
    img_rows, img_cols = X_test.shape[1], X_test.shape[2]
    if K.image_dim_ordering() == 'th':
        X_test = np.swapaxes(X_test,1,3)
        X_test = np.swapaxes(X_test,2,3)
        input_shape = (X_test.shape[1], img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, X_test.shape[3])
    X_test = X_test.astype('float32')
    X_test /= 255
    results = model.predict(X_test)

    np.save(outfilename,results[0,:])
    return results[0,:]
    
def getNeighbors(features):
    distances, indices = tree.query(features,3)
    idsToShow = [photo_ids[i] for i in indices]
    return idsToShow

def getDogData(dogIds):
    dogdata = []
    for dog in dogIds:
        di = dict()
        di["name"] = dog_info[dog]["name"]
        di["desc"] = dog_info[dog]["description"]
        dogdata.append(di)
    return dogdata
