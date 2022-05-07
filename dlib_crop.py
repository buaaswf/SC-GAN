# _*_ coding:utf-8 _*_

import numpy as np
import cv2
import dlib
import os


def get_landmarks(img, detector, predictor):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rects = detector(img_gray, 0)
    #assert(len(rects) == 1)
    if len(rects) > 0:
        results = predictor(img, rects[0])
        landmarks = np.matrix([[p.x, p.y] for p in results.parts()])
        return landmarks
    else:
        return []


def draw_landmarks(img, landmarks):
    land_img = img.copy()
    for idx, point in enumerate(landmarks):
        position = (point[0, 0], point[0, 1])
        #print(idx, position)
        #cv2.circle(land_img, position, 3, color=(0, 255, 0))
        cv2.putText(land_img, str(idx + 1), position, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)

    return land_img

def crop_2eyes(img, landmarks):
    if len(landmarks) <= 0:
        return None
    '''
      two eyes rect
      19 20 21 24 25 26
     1                 17
              29
    '''
    x1 = landmarks[0, 0]
    x2 = landmarks[16, 0]
    y1 = landmarks[18, 1]
    y1 = min(y1, landmarks[19, 1])
    y1 = min(y1, landmarks[20, 1])
    y1 = min(y1, landmarks[23, 1])
    y1 = min(y1, landmarks[24, 1])
    y1 = min(y1, landmarks[25, 1])
    y2 = landmarks[28, 1]

    crop_img = img[y1: y2, x1: x2, :]
    return crop_img

def crop_mouth(img, landmarks):
    if len(landmarks) <= 0:
        return None
    '''
      mouth rect
            50 51 52 53 54
              62 63 64
     49 61    68 67 66      65 55
            60 59 58 57 56
    '''
    x1 = landmarks[48, 0]
    x2 = landmarks[54, 0]
    y1 = landmarks[50, 1]
    y2 = landmarks[57, 1]

    h = y2 - y1
    w = x2 - x1

    H, W, C = img.shape

    x1 = max(0, x1 - w / 4)
    x2 = min(W, x2 + w / 4)

    y1 = max(0, y1 - h / 3)
    y2 = min(H, y2 + h / 3)

    crop_img = img[y1: y2, x1: x2, :]
    return crop_img


def main():
    src_dir = "/home/liuliang/Desktop/dataset/liuji/CelebA/Img/img_align_celeba"
    des_dir = "./1_crop/"

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    img_ids = os.listdir(src_dir)
    img_ids.sort()
    cnt = len(img_ids)
    print("Images: {}".format(cnt))

    if not os.path.exists(des_dir):
        os.makedirs(des_dir)

    for i in range(cnt):
        img_id = img_ids[i]
        print("[{}/{}] {}".format(i, cnt, img_id))

        src_path = os.path.join(src_dir, img_id)
        des_path = os.path.join(des_dir, img_id)

        img = cv2.imread(src_path)

        landmarks = get_landmarks(img, detector, predictor)
        #print(landmarks)
        
        #res_img = draw_landmarks(img, landmarks)
        #res_img = crop_2eyes(img, landmarks)
        res_img = crop_mouth(img, landmarks)
        
        if res_img is  not None:
            cv2.imwrite(des_path, res_img)
            #cv2.imwrite(des_path[:-4] + '_origin.jpg', img)


if __name__ == "__main__":
    main()
