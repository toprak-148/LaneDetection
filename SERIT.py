import cv2
import numpy as np
import math
sapma = 100
kernel = np.ones((5,5),dtype=np.uint8)
def regionOfInterest(image):
    width , height = image.shape[:2]
    x1 = 1
    y1 = 290
    x2 = 140 
    y2 = 210
    x3 = 300
    y3 = 210
    x4 = 400
    y4 = 290
    polygon = np.array([[(x1,y1),(x2,y2),(x3,y3),(x4,y4)]])
    return polygon

def cropImage(image,matris):
    x ,y = image.shape[:2]
    mask = np.zeros(shape=(x,y),dtype=np.uint8)
    mask = cv2.fillPoly(mask,matris,255)
    mask = cv2.bitwise_and(image, image,mask=mask)
    return mask

def thresholding(image,lowerThreshold,upperThreshold):
    grayImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret , thresImage = cv2.threshold(grayImage, lowerThreshold, upperThreshold, cv2.THRESH_BINARY)
    thresImage = cv2.dilate(thresImage, kernel = kernel)
    thresImage = cv2.erode(src = thresImage, kernel = kernel)
    blurImage = cv2.medianBlur(thresImage,5)
    return blurImage

def canny(blurImage,lowerCanny,upperCanny):
    cannyImage = cv2.Canny(blurImage,lowerCanny,upperCanny)
    return cannyImage



def drawPolyLines(image,matris):
        x,y = image.shape[:2]
        dst = np.array([[matris[0][1, 0],  matris[0][1, 1]],
                    [ matris[0][0, 0],  matris[0][0, 1]],
                    [ matris[0][3, 0], matris[0][3, 1]],
                    [ matris[0][2, 0],  matris[0][2, 1]]], np.int32)
        cv2.polylines( image, [dst], True, (0,255,0),4)
        return image
    
    
def averageSlopeIntercept(image,lineSegments):
    numberOfEndLine = 0
    rightLength = 0
    leftLength = 0
    l = 0
    boundary = 1/12#ayarlanacak
    
    laneLines= []
    
    if lineSegments is None:
        print("Çizgi tespit edilemedi")
        return laneLines
    
    height,width = image.shape[:2]
    leftFit = []
    rightFit = []
    leftRegionBounday = width * (1-boundary) # tekrar ayarlanacak
    rightRegionBounday = width *  boundary # tekrar ayarlanacak
    
    for line_segment in lineSegments:
        for x1,y1,x2,y2 in line_segment:
            if x1==x2:
                print("yatay çizgi mevcut değil")
                continue
            
            fit = np.polyfit((x1,x2), (y1,y2),1)
            slope = fit[0]
            intercept = fit[1]
            
            endLineLength = x2 - 1 
            
            if endLineLength < 0:
                endLineLength *= -1
                
            cizgiUzunluk = y2 - y1
            if cizgiUzunluk < 0 :
                cizgiUzunluk *= -1
                
            if slope < -0.3:
                if x1 < leftRegionBounday and x2 < leftRegionBounday:
                    leftFit.append((slope,intercept))
                    if cizgiUzunluk > rightLength and cizgiUzunluk > 50:
                        print("kesikli sag")
                        rightLength = cizgiUzunluk
                        print("sag uzunluk : ",rightLength)
                        rightLength = 0.0
            
            if slope > 0.3:
                if x1 > rightRegionBounday and x2 > rightRegionBounday : 
                    rightFit.append((slope,intercept))
                    if cizgiUzunluk > leftLength and cizgiUzunluk > 50:
                        print("kesikli sol")
                        leftLength = cizgiUzunluk
                        print("sol uzunluk",leftLength)
                        leftLength = 0.0
          
            if endLineLength > 160:
                numberOfEndLine += 1 
                if len(laneLines) == 2 or len(laneLines) == 1:
                    l += 1
                    if l > 45:
                        numberOfEndLine = 0
                        
    leftFitAverage = np.average(leftFit,axis = 0)
    if len(leftFit) > 0:
        laneLines.append(makePoints(image, leftFitAverage))
        
    rightFitAverage = np.average(rightFit,axis = 0)
    if len(rightFit) > 0:
        laneLines.append(makePoints(image,rightFitAverage))

    return laneLines


def makePoints(image,lines):
    height,width = image.shape[:2]
    
    if lines is not None:
        slope , intercept = lines
        abs(slope)
        
        if slope != 0:
            y1 = height 
            y2 = int(y1 * ( 3 / 5 ) )
            x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
            x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
        
            return [[x1,y1,x2,y2]]
      
        else:
            print("eğim 0 ")
    
    else:
        print("koordinat bulunamadı")
        return -90
    
def computeSteeringAngle(image, lines, cameraHas=-0.04):
    height, width = image.shape[:2]
    steering_angle = 0
    
    if lines is None or len(lines) == 0:
        return 100  # Hiç çizgi tespit edilemezse varsayılan değer
    
    leftLines = []
    rightLines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue
        fit = np.polyfit((x1, x2), (y1, y2), 1)
        slope = fit[0]
        
        if slope < -0.3:
            leftLines.append((x1, y1, x2, y2))
        elif slope > 0.3:
            rightLines.append((x1, y1, x2, y2))
    
    if len(leftLines) == 0 and len(rightLines) == 0:
        return 100  # Hiç geçerli çizgi tespit edilemezse varsayılan değer
    
    if len(rightLines) == 0:
        left_lines_mean = np.mean(leftLines, axis=0)
        left_mid_point = ((left_lines_mean[0] + left_lines_mean[2]) / 2, (left_lines_mean[1] + left_lines_mean[3]) / 2)
        mid_point_deltaX = left_mid_point[0] * (1+cameraHas)
        mid_point_deltaY = left_mid_point[1] *(1+cameraHas)
        lane_angle_radians = math.atan2(mid_point_deltaY, mid_point_deltaX)
        steering_angle = math.degrees(lane_angle_radians)
        steering_angle += 85
        steering_angle = math.ceil(steering_angle)
        if steering_angle > 180:
            steering_angle = 140
            
    elif len(leftLines) == 0:
        right_lines_mean = np.mean(rightLines, axis=0)
        right_mid_point = ((right_lines_mean[0] + right_lines_mean[2]) / 2, (right_lines_mean[1] + right_lines_mean[3]) / 2)
        mid_point_deltaX = right_mid_point[0] *(1+cameraHas)
        mid_point_deltaY = right_mid_point[1] *(1+cameraHas)
        lane_angle_radians = math.atan2(mid_point_deltaY, mid_point_deltaX)
        steering_angle = math.degrees(lane_angle_radians)
        steering_angle = math.ceil(steering_angle)
        if steering_angle > 180:
            steering_angle = 30
            
    elif 0 < len(leftLines) and 0 < len(rightLines):
        left_lines_mean = np.mean(leftLines, axis=0)
        right_lines_mean = np.mean(rightLines, axis=0)
        _, _, left_x2, _ = left_lines_mean
        _, _, right_x2, _ = right_lines_mean
        mid = int((width / 3.2) * (1 + cameraHas))
        x_offset = (left_x2 + right_x2) / 2 - mid
        y_offset = int(height / 2)
        
        angle_to_mid_radian = math.atan(x_offset / y_offset)
        angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
        steering_angle = angle_to_mid_deg + 90
        
    else:
        steering_angle = 100
    
    return steering_angle


def displayHeadingLine(image,steeringAngle):
    height,width = image.shape[:2]
    
    
    steeringAngleRadian = math.radians(steeringAngle)
    steeringAngleTan = math.tan(steeringAngleRadian)
    
    x1 = int(width / 3.2)
    y1 = height
    if steeringAngle != 0:
        x2 = int(x1- ((height / 2)/steeringAngleTan))
    else:
        x2 = x1 
    y2 = int(y1*(3/5))
    cv2.line(image,(x1,y1),(x2,y2),(255,0,255),5)
    return image

            