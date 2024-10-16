import cv2 
import numpy as np
import pyautogui
import SERIT as srt
 

# while True:

#     screenshot = pyautogui.screenshot()
#     screenshot.save("C:/Users/b_erd/Desktop/B/a.png")
#     img = cv.imread("C:/Users/b_erd/Desktop/B/a.png")

#     matris = ovat.cropMatris(img)
#     crop = ovat.cropImage(img, matris)
#     filter_image = ovat.Filter(crop)
#     line_segments = cv.HoughLinesP(filter_image, 1, np.pi / 180, 10, np.array([]), minLineLength=3, maxLineGap=70)
#     lane_lines = ovat.average_slope_intercept(img, line_segments)
#     steering_angle = ovat.compute_lane_angle(img, lane_lines)
#     angle_string = str(steering_angle)

#     with open("C:/Users/b_erd/Desktop/a.txt", "w") as file:
#         file.write(angle_string)

#     if lane_lines is not None:
#         for line in lane_lines:
#             if line is not None:
#                 x1, y1, x2, y2 = line[0]
#                 cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 9)

#     heading_line_image = ovat.display_heading_line(img, steering_angle)
#     image_with_lines = cv.addWeighted(img, 0.8, heading_line_image, 1, 1) 
#     img = ovat.drawPolyLines(img, matris)
#     image_with_lines = ovat.drawPolyLines(image_with_lines, matris)
#     cv.namedWindow("Image Width Lines", cv.WINDOW_NORMAL)
#     cv.imshow("Image Width Lines", image_with_lines)

#     if cv.waitKey(1000) & 0xFF == ord('q'):
#         break

# cv.destroyAllWindows()

cap = cv2.VideoCapture("SimiParkur.mp4")

while cap.isOpened():
    
    success,image = cap.read()
    cv2.imshow("First Frame", image)
    matris = srt.regionOfInterest(image)
    cropImage = srt.cropImage(image, matris)
    thresImage = srt.thresholding(cropImage,lowerThreshold=200,upperThreshold= 255)
    filterImage = srt.canny(thresImage,lowerCanny = 0,upperCanny = 255)
    lineSegments = cv2.HoughLinesP(filterImage, 1,np.pi/180.0, 20,np.array([]),minLineLength=1,maxLineGap=40)
    laneLines = srt.averageSlopeIntercept(image, lineSegments)
    steeringAngle = srt.computeSteeringAngle(image, laneLines)
    
    
    
    if laneLines is not None:
        for line in laneLines:
            if line is not None:
                x1,y1,x2,y2 = line[0]
                cv2.line(image, (x1,y1), (x2,y2), (0,0,255),10)
                
                
    print(steeringAngle)
    headingLineImage = srt.displayHeadingLine(image, steeringAngle)
    imageWithLines = cv2.addWeighted(image,0.8,headingLineImage,beta = 1, gamma = 1)
    imageWithLines = srt.drawPolyLines(image, matris)
        
    cv2.imshow("Image", image)
    cv2.imshow("CropImage", cropImage)    
    cv2.imshow("thresImage",thresImage)
    cv2.imshow("FilterImage", filterImage)
    cv2.imshow("Combo Image", imageWithLines)    
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    