import numpy as np
import pandas as pd
import cv2

eyes_cascade = cv2.CascadeClassifier('./Train/third-party/frontalEyes35x16.xml')
nose_cascade = cv2.CascadeClassifier('./Train/third-party/Nose18x15.xml')


Before = cv2.imread('./Test/Before.png')
sunglasses = cv2.imread('./Train/glasses.png', -1)
mustache = cv2.imread('./Train/mustache.png', -1)

image = np.copy(Before)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

eyes = eyes_cascade.detectMultiScale(gray, 1.3, 5)
eyes = sorted(eyes, key=lambda f: f[2]*f[3])
for eye in eyes[-1:]:
    x, y, w, h = eye
    sunglass = np.copy(sunglasses)
    w = int(np.ceil(w * 650 / 483))
    h = int(np.ceil(h*350/221))
    sunglass = cv2.resize(sunglass, (w, h))
    disty = int(np.ceil(image.shape[0] * 20 / 1050))
    distx = int(np.ceil(image.shape[1] * 80 / 1050))
    for i in range(h):
        for j in range(w):
            if sunglass[i, j, -1] != 0:
                image[y+i-disty, x+j-distx] = sunglass[i, j, :-1]

noses = nose_cascade.detectMultiScale(gray, 1.3, 5)
noses = sorted(noses, key=lambda f: f[2]*f[3])
for nose in noses[-1:]:
    x, y, w, h = nose
    w = int(np.ceil(w * 700 / 483))
    h = int(np.ceil(h * 350 / 290))
    m = np.copy(mustache)
    m = cv2.resize(m, (w, h))
    d = int(np.ceil(image.shape[0]*48/1050))
    dist = int(np.ceil(image.shape[1]*20/1050))
    for i in range(h):
        for j in range(w):
            if m[i, j, -1] != 0:
                image[y+i+d, x+j-dist] = m[i, j, :-1]


cv2.imshow("After", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('After.png', image)

image_ = image
image_ = image_.reshape((image.shape[0]*image.shape[1], 3))

d = {'Channel 1': image_[:, 2],
     'Channel 2': image_[:, 1],
     'Channel 3': image_[:, 0]}

df = pd.DataFrame(data=d)
df.to_csv(r'Output.csv', index=False)
