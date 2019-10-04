import cv2
import numpy as np 
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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
