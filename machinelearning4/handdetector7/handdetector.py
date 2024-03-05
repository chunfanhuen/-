import cv2 as cv
import mediapipe as mp


class HandDetector:
    '''
    手势识别类
    '''

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
            self.mode,
            self.max_hands,
            self.detection_con,
            self.track_con

        )

    def find_hands(self, img):
        # 需要把BGR格式转换为RGB格式 才能传入process方法中
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # 检测是否有手势（注意参数必须是RGB格式的图片）
        self.results = self.hands.process(imgRGB)
        # 只有检测到手 才进行处理
        if self.results.multi_hand_landmarks:
            # 遍历每只手
            for handlms in self.results.multi_hand_landmarks:
                # 绘制手势
                # 参数1 image 在哪张图片上进行绘制 需要为3通道的RGB数组
                # 参数2 landmark_list 手势列
                # 参数3 connections 手指关节列表的连接
                mp.solutions.drawing_utils.draw_landmarks(
                    imgRGB,
                    handlms,
                    mp.solutions.hands.HAND_CONNECTIONS
                )

        # 把绘制手势后的RGB格式转换回BGR格式
        img = cv.cvtColor(imgRGB, cv.COLOR_RGB2BGR)

        return img

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
