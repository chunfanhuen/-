import cv2 as cv
import mediapipe as mp

class HandDetector:
    '''
    手势识别类
    '''
    def __init__(self, mode=False,max_hands=2,detection_con=0.5,track_con=0.5):
        '''
        手势识别初始化
        :param mode: 是否静态图片。默认为False
        :param max_hands: 最多识别几只手。默认为2
        :param detection_con: 最小检测置信度值。默认为0.5
        :param track_con: 最小追踪置信度值。默认为0.5
        '''
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        self.hands = mp.solutions.hands.Hands(
            mode,
            max_hands,
            detection_con,
            track_con
        )