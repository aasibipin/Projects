import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
from statistics import mean

cap = cv2.VideoCapture(0)

template = cv2.imread('ieee2.PNG')
template_gray = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)

w,h = template_gray.shape[::-1]
locations = []

while True:
    cv2.waitKey(70)
    _, frame = cap.read()
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(gray_frame, template_gray, cv2.TM_CCOEFF_NORMED)
    THRES = 0.43
    loc = np.where(res>=THRES)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,255,0),2)
        locations.append(pt)

    cv2.imshow("frame", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
    '''
ord('q') returns the Unicode code point of q
returns a 32-bit integer corresponding to the pressed key
& 0xFF is a bit mask which sets the left 24 bits to zero, because ord() 
 returns a value betwen 0 and 255, since your keyboard only has a limited character set
'''

plt.plot(*zip(*locations))
plt.title('Displacement')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.show()

cv2.destroyAllWindows()
cap.release()