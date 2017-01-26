from subprocess import call
import matplotlib as plt
import cv2
import csv

def processImage(filepath,imagefilename):
    shrunkImageName, scaleRatio = shrinkImage(filepath,imagefilename)
    clippedImageName, boundingBox = runDarknet(filepath,shrunkImageName)
    boundedImageName = drawBoundingBoxOnImage(filepath,imagefilename,boundingBox,scaleRatio)
    return boundedImageName

def shrinkImage(filepath,imagefilename):
    #set our target height
    targetHeight = 600.0

    #read in original image
    imagename = filepath+"/"+imagefilename
    img = cv2.imread(imagename)

    #check that the image is larger than our target size, if not return
    if img.shape[0] < targetHeight: return imagefilename, 1.0

    #calculate aspect ratio and target width to maintain image dimensions when shrinking
    aspectRatio = float(img.shape[0])/float(img.shape[1])
    targetWidth = targetHeight / aspectRatio
    
    #resize image
    imgout = cv2.resize(img, (int(targetWidth), int(targetHeight)), cv2.INTER_LINEAR)

    #set output image name
    filebase = imagefilename.rsplit('.', 1)[0]
    fileext = imagefilename.rsplit('.', 1)[1]
    newimagefilename = filebase+"_shrunk."+fileext

    #write the image to same directory, but with an extra tag
    cv2.imwrite(filepath+"/"+newimagefilename,imgout)

    #return new filename and ratio for showing bounding box
    return newimagefilename, img.shape[0] / targetHeight

def runDarknet(filepath,imagefilename):
    #base command to run darknet
    command=["darknet","detect","./cfg/yolo.cfg","./yolo.weights"]
    #add file to search
    command.append(filepath+"/"+imagefilename)
    #add output prefices
    command.append("-prefix")
    filebase=imagefilename.rsplit('.', 1)[0]
    fileext=imagefilename.rsplit('.', 1)[1]

    prefix = filepath+"/"+filebase+"_result"
    command.append(prefix)

    #run darknet
    call(command)

    #read darknet output and find "best" dog
    mImg = None
    mImgScore = 0
    mBox = (0,0,0,0)
    with open(prefix+".csv", 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            #only look at dogs that were detected
            if row[-1] != "dog":continue
            #choose the "best" probability dog
            score = float(row[-2])
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
    ofname = filebase+"_clipped."+fileext
    cv2.imwrite(filepath+"/"+ofname,mImg)
    #return bounding box coordinates and new filename
    return ofname,mBox

def drawBoundingBoxOnImage(filepath,imagename,boundingBox,scaleFactor):
    filebase=imagename.rsplit('.', 1)[0]
    fileext=imagename.rsplit('.', 1)[1]

    img = cv2.imread(filepath+"/"+imagename)
    x1 = int(boundingBox[0] * scaleFactor)
    y1 = int(boundingBox[1] * scaleFactor)
    x2 = int((boundingBox[0] + boundingBox[2])*scaleFactor)
    y2 = int((boundingBox[1] + boundingBox[3])*scaleFactor)
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),5)

    outfilename = filebase+"_bounding."+fileext

    cv2.imwrite(filepath+outfilename,img)

    return outfilename
