
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from functools import reduce
from scipy.interpolate import interp1d
import math
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def extract_bboxes(fused):
    """Compute bounding boxes from masks.
    mask: [height, width]..
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    mask = cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY)
    mask[mask < 40] = 0
    mask[mask >= 40] = 1
    mask = mask.reshape(256, 256, 1)
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)

def getContours(npImage, overlay_img, realHeight, realWidth, unit, confidence, angle_th=30):
    # load the image, convert it to grayscale, and blur it slightly
    image = npImage.copy()#cv2.imread(imagePath)
    # image size
    imgHeight = image.shape[0]
    imgWidth = image.shape[1]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 50, 80)
    
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(image=overlay_img, contours=cnts[0], contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    cnts = imutils.grab_contours(cnts)
    
    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetricHeight = realHeight/imgHeight
    pixelsPerMetricWidth = realWidth/imgWidth
    #draw bounding box
    y1, x1, y2, x2 = extract_bboxes(npImage)[0]
    cv2.rectangle(overlay_img,(x1,y1),(x2, y2),(0,255,0),2)
    # loop over the contours individually
    
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 100:
            continue
        # compute the rotated bounding box of the contour
        orig = overlay_img.copy()
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = perspective.order_points(box)

        # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # draw lines between the midpoints
        #cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                 #(255, 0, 0), 1)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                 (255, 0, 0), 1)
        

        top_p = min([(int(tlblX), int(tlblY)), (int(trbrX), int(trbrY))], key=lambda x : x[1])
        bot_p = max([(int(tlblX), int(tlblY)), (int(trbrX), int(trbrY))], key=lambda x : x[1])
        D_ad = ((top_p[1] - bot_p[1]) ** 2 + (top_p[0] - bot_p[0])**2) ** 0.5 + 1e-7
    
        P1 = min(top_p, bot_p, key=lambda x:x[0])
        P2 = max(top_p, bot_p, key=lambda x:x[0])
        slope = (P1[1] - P2[1]) / (P2[0] - P1[0])
        cat = ''
        angle = 0        
        if slope > 0:
            angle = np.arccos((top_p[0] - bot_p[0])/D_ad) * 180 / math.pi
            cv2.putText(orig, "angle={:.1f}".format(angle),
            (max(top_p[0]-100, 0), top_p[1] + 15), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (0, 0, 255), 1)
        else:
            angle = np.arccos((bot_p[0] - top_p[0])/D_ad) * 180 / math.pi  
            cv2.putText(orig, "angle={:.1f}".format(angle),
            (top_p[0], top_p[1] + 15), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (0, 0, 255), 1)
        

        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        length = cv2.arcLength(c, True) / 2. * pixelsPerMetricWidth
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    
        # width
        mask = gray.copy()
        mask[mask < 40] = 0
        width = cv2.countNonZero(mask[cY][:])
        right_most_x = np.max(np.nonzero(mask[cY][:]))
        left_most_x = np.min(np.nonzero(mask[cY][:]))
        cv2.line(orig, (int(left_most_x), int(cY)), (int(right_most_x), int(cY)),
                 (255, 0, 0), 1)
        width *= pixelsPerMetricWidth 
        # compute the size of the object
        dimA = dA * pixelsPerMetricHeight
        dimB = dB * pixelsPerMetricWidth
        if angle < angle_th:
            cat +='H'
            cv2.putText(orig, "L={:.1f}".format(length) + unit,
            (int(tltrX), int(tltrY) + 40), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (0, 0, 255), 1)
            cv2.putText(orig, "W={:.1f}".format(width) + unit,
            (int(tltrX), int(tltrY) + 55), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (0, 0, 255), 1)
        else:
            cat += 'V'
            cv2.putText(orig, "L={:.1f}".format(length) + unit,
            (int(tltrX), int(tltrY)), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (0, 0, 255), 1)
            cv2.putText(orig, "W={:.1f}".format(width) + unit,
            (int(tltrX), int(tltrY) + 15), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (0, 0, 255), 1)
        if slope > 0:
            cat += 'L'
        else:
            cat += 'R'
        cv2.putText(orig, "cat="+cat + " Crack percentage={:.2f}".format(confidence.item()), (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX,  0.45, (36,255,12), 1)
        
        return orig


