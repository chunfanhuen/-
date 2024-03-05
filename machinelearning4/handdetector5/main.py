import cv2 as cv
import numpy as np
# from 包名.文件名 import 类(*)
from handdetector import *


def main():
    num = 0
    # 打开摄像头
    # 参数为字符串时，表示视频文件路径
    # 参数为整数时，表示摄像头ID。0（内置摄像头）、1（外置摄像头）
    cap = cv.VideoCapture(0)
    # 创建手势识别对象
    detector = HandDetector()
    while True:
        # 将视频帧读取到img数据中
        # 返回值1 flag 如果还有下一帧 该值为True 否则为False（对于读取摄像头来说，该值永远为True）
        # 返回值2 img 该视频帧的图片数据
        flag, img = cap.read()
        # 摄像头左右会颠倒 需要使用下面这句才能翻转过来
        img = cv.flip(img, 1)
        # 检测手势
        img = detector.find_hands(img)
        # 获取手势数据
        lmslist = detector.find_positions(img)
        if len(lmslist) > 0:
            print('lmslist:', lmslist)
            print('lmslist shape:', np.array(lmslist).shape)
        # 显示画面
        cv.imshow('img', img)
        # 等待用户按键
        key = cv.waitKey(1)
        # 退出功能
        if key == ord('q'):
            break
        # 自拍功能
        elif key == ord('s'):
            num += 1
            cv.imwrite('face%d.jpg' % num, img)

    cap.release()
    cv.destroyAllWindows()
