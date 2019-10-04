import cv2
import numpy as np 
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #hsv hue set value
    lower_red = np.array([1,0,0])
    upper_red = np.array([6,225,255])
    kernal = np.ones((10,10),np.uint8)

    mask = cv2.inRange(hsv, lower_red, upper_red)   #Mask = 1 or = 0 
    opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernal)
    closing = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernal)
    
    res = cv2.bitwise_and(frame, frame, mask=mask)
    temp1 = cv2.bitwise_and(res, res, mask=opening)
    final = cv2.bitwise_and(temp1, temp1, mask=closing)

    median = cv2.medianBlur(final,15)

    cv2.imshow('median',median)
    cv2.imshow('opening',opening)
    cv2.imshow('closing',closing)
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
    '''
ord('q') returns the Unicode code point of q
returns a 32-bit integer corresponding to the pressed key
& 0xFF is a bit mask which sets the left 24 bits to zero, because ord() 
 returns a value betwen 0 and 255, since your keyboard only has a limited character set
'''

cv2.destroyAllWindows()
cap.release()
