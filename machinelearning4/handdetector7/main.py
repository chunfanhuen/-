import cv2 as cv
from handdetector import *
import numpy as np


def main():
    num = 0
    # 打开摄像头
    # 参数为字符串，表示视频文件的路径
    # 参数为整数时，表示是摄像头id 0（内置） 1（外置）
    cap = cv.VideoCapture(0)
    # 创建手势对象
    detector = HandDetector()
    # 指尖id列表
    tip_ids = [4, 8, 12, 16, 20]
    # 6张手势图片，分别代表0~5
    finger_img_list = [
        'fingers/0.png',
        'fingers/1.png',
        'fingers/2.png',
        'fingers/3.png',
        'fingers/4.png',
        'fingers/5.png',
    ]
    finger_list = []
    for fi in finger_img_list:
        i = cv.imread(fi)
        finger_list.append(i)
    while True:
        flag, img = cap.read()
        # 反转镜像
        img = cv.flip(img, 1)
        img = detector.find_hands(img)
        lmslist = detector.find_positions(img)
        if len(lmslist) > 0:
            # print("lmslist", lmslist)
            # print("lmslist shape", np.array(lmslist).shape)
            fingers = []  # 记录每个手指的转态（0 收起，1 打开）
            for tip in tip_ids:
                # 找到每个指尖的位置
                x, y = lmslist[tip][1], lmslist[tip][2]
                cv.circle(img, (x, y), 10, (0, 255, 0), cv.FILLED)
                if tip == 4:  # 大拇指
                    # 根据食指（8）和中指（12）的位置判断左手右手
                    if lmslist[8][1] < lmslist[12][1]:  # 右手
                        if lmslist[tip][1] < lmslist[tip - 1][1]:
                            fingers.append(1)  # 大拇指打开
                        else:
                            fingers.append(0)
                    else:  # 左手
                        if lmslist[tip][1] > lmslist[tip - 1][1]:
                            fingers.append(1)  # 大拇指打开
                        else:
                            fingers.append(0)
                else:  # 其他手指
                    if lmslist[tip][2] < lmslist[tip - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
            print(fingers)
            cnt = fingers.count(1)
            print('总共有%d个手指打开' % cnt)
            # 找到对应的手势图片并展示
            finger_img = finger_list[cnt]
            w, h, c = finger_img.shape  # 得到展示图片的宽和高
            img[0:2, 0:h] = finger_img  # 展示在原图img左上角的位置
        cv.imshow('img', img)
        key = cv.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            num += 1
            cv.imwrite("face%d.jpg" % num, img)
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
