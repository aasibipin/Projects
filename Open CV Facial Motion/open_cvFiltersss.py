import cv2
import numpy as np 
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([100,100,0])
    upper_red = np.array([120,225,255])
    kernal = np.ones((10,10),np.uint8)

    mask = cv2.inRange(hsv, lower_red, upper_red)   #Mask = 1 or = 0 
    
    res = cv2.bitwise_and(frame, frame, mask=mask)


    cv2.imshow('mask',mask)
    cv2.imshow('res',res)


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