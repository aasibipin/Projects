import cv2
import numpy as np 
cap = cv2.VideoCapture(0)
template = cv2.imread('RSU.PNG')
template_gray = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)

w,h = template_gray.shape[::-1]

while True:
    _, frame = cap.read()
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(gray_frame, template_gray, cv2.TM_CCOEFF_NORMED)
    thres = 0.5
    loc = np.where(res>=thres)
    
    for pt in zip(*loc[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,255,0),2)

    cv2.imshow("frame", frame)


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