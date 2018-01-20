# coding: utf8
# opencv提取特征
import numpy as np
import cv2

def sift_desc(rgb_name, rectangle=None):    # rectangle=[x1,y1,x2,y2]
    img = cv2.imread(rgb_name, 0)
    sift = cv2.xfeatures2d.SIFT_create()
    if rectangle:
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.rectangle(mask, (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3]), (255, 255, 255), -1)
        kp, desc = sift.detectAndCompute(img, mask)
    else:
        kp, desc = sift.detectAndCompute(img, None)
    return kp, desc   # kp: 关键点 desc: 特征描述子
#################sift使用########################
# kp1, desc1 = sift_desc(rgb1)
# kp_img = cv2.drawKeypoints(img1, kp, None)
# plt.imshow(kp_img)
# kp2, desc2 = sift_desc(rgb2)
# bf = cv2.BFMatcher()
# matches = bf.match(desc1, desc2)
# matches = sorted(matches, key=lambda x:x.distance)
# img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None)
# plt.imshow(img3)
# plt.show()
################################################

# corr = abs(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)) 求相关性
# mask部分可参照sift_desc
def calcHist_rgb(image_name, mask=None):
    image = cv2.imread(image_name)
    hist_b = cv2.calcHist([image], [0], mask, [256], [0, 256])
    cv2.normalize(hist_b, hist_b, 0.0, 1.0, cv2.NORM_MINMAX)
    hist_g = cv2.calcHist([image], [1], mask, [256], [0, 256])
    cv2.normalize(hist_g, hist_g, 0.0, 1.0, cv2.NORM_MINMAX)
    hist_r = cv2.calcHist([image], [2], mask, [256], [0, 256])
    cv2.normalize(hist_r, hist_r, 0.0, 1.0, cv2.NORM_MINMAX)
    return np.array(list(hist_r)+list(hist_g)+list(hist_b)).flatten()
def calcHist_hsv(image_name, mask=None):
    image = cv2.imread(image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([image], [0], mask, [180], [0, 180])
    cv2.normalize(hist_h, hist_h, 0.0, 1.0, cv2.NORM_MINMAX)
    hist_s = cv2.calcHist([image], [1], mask, [256], [0, 256])
    cv2.normalize(hist_s, hist_s, 0.0, 1.0, cv2.NORM_MINMAX)
    hist_v = cv2.calcHist([image], [2], mask, [256], [0, 256])
    cv2.normalize(hist_v, hist_v, 0.0, 1.0, cv2.NORM_MINMAX)
    return np.array(list(hist_h)+list(hist_s)+list(hist_v)).flatten()

