import cv2 as cv
import mediapipe as mp


class HandDetector:

    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
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


def find_positions(self, img, hand_no=0):
    '''
    获取手势数据
    :param img: 视频帧图片
    :return:
    '''
    self.lmslist = []
    # 只有检测到手 才进行处理
    if self.results.multi_hand_landmarks:
        # 获取第几只手的数据
        hand = self.results.multi_hand_landmarks[hand_no]
        # 遍历这只手的关节点的 id 和 landmark
        for id, lm in enumerate(hand.landmark):
            # 得到整张图片的大小分辨率
            h, w, c = img.shape
            # 根据landmark的x和y的比例得到该关节点具体的坐标
            cx, cy = int(lm.x * w), int(lm.y * h)
            # 把关节点的坐标保存到lmslist二维列表中
            self.lmslist.append([id, cx, cy])

    # lmslist的shape为21*3，数据为：
    # [[0, 77, 360],
    #  [1, 149, 344],
    #  ...,
    #  [20, 130, 156]]
    return self.lmslist
