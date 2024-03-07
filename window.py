# 可视化界面
# 应该在界面启动的时候就将模型加载出来，设置tmp的目录来放中间的处理结果
import shutil
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
import torch
import os.path as osp
from model_train import SELFMODEL
import numpy as np
from torch import nn
from torchutils import get_torch_transforms
from PIL import Image
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class MainWindow(QTabWidget):
    # 基本配置不动，然后只动第三个界面
    def __init__(self):
        # 初始化界面
        super().__init__()
        self.setWindowTitle('水色图像识别系统')
        self.resize(1200, 800)
        self.setWindowIcon(QIcon("image/window_ui/logo.png"))
        # 图片读取进程
        self.output_size = 480
        self.img2predict = ""
        self.origin_shape = ()

        self.model_path = "./checkpoints/resnet50d_pretrained_224/resnet50d_10epochs_accuracy0.99970_weights.pth"  # todo  模型路径
        # “一类”、“二类”、“三类”、“四类”
        self.classes_names = ['第一类水质，浅绿色，水质：优', '第二类水质，灰蓝色，水质：良', '第三类水质，黄褐色，水质：较差', '第四类水质，茶褐色，水质：差']  # todo 类名
        self.img_size = 224  # todo 图片大小
        self.model_name = "resnet50d"  # todo 模型名称
        self.num_classes = len(self.classes_names)  # todo 类别数目
        # 加载网络
        model = SELFMODEL(model_name=self.model_name, out_features=self.num_classes, pretrained=False)
        weights = torch.load(self.model_path,
                             map_location=torch.device('cpu'))
        model.load_state_dict(weights)
        model.eval()
        model.to(device)
        self.model = model

        # 加载数据处理
        data_transforms = get_torch_transforms(img_size=self.img_size)
        # train_transforms = data_transforms['train']
        self.valid_transforms = data_transforms['val']
        self.initUI()

    '''
    ***界面初始化***
    '''
    def initUI(self):
        # 图片检测子界面
        font_title = QFont('楷体', 16)
        font_main = QFont('楷体', 14)
        # 图片识别界面, 两个按钮，上传图片和显示结果
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()
        img_detection_title = QLabel("图片识别功能")
        img_detection_title.setFont(font_title)
        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()
        self.left_img = QLabel()
        self.right_img = QLabel()
        self.left_img.setPixmap(QPixmap("image/window_ui/upload.png"))
        self.right_img.setPixmap(QPixmap("image/window_ui/right.png"))
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        mid_img_layout.addWidget(self.left_img)
        mid_img_layout.addStretch(0)
        # mid_img_layout.addWidget(self.right_img)
        mid_img_widget.setLayout(mid_img_layout)
        up_img_button = QPushButton("上传图片")
        det_img_button = QPushButton("识别")
        up_img_button.clicked.connect(self.upload_img)
        det_img_button.clicked.connect(self.detect_img)
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        #
        self.rrr = QLabel("等待识别")
        self.rrr.setFont(font_main)
        up_img_button.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"
                                    "QPushButton{background-color:rgb(48,124,208)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:5px 5px}"
                                    "QPushButton{margin:5px 5px}")
        det_img_button.setStyleSheet("QPushButton{color:white}"
                                     "QPushButton:hover{background-color: rgb(2,110,180);}"
                                     "QPushButton{background-color:rgb(48,124,208)}"
                                     "QPushButton{border:2px}"
                                     "QPushButton{border-radius:5px}"
                                     "QPushButton{padding:5px 5px}"
                                     "QPushButton{margin:5px 5px}")
        img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(self.rrr)
        # img_detection_layout.addWidget(self.c1)
        img_detection_layout.addWidget(up_img_button)
        img_detection_layout.addWidget(det_img_button)
        img_detection_widget.setLayout(img_detection_layout)

        # todo 关于界面
        '''
        *** 关于界面 ***
        '''
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel('欢迎使用水色图像识别识别系统')  # todo 修改欢迎词语
        about_title.setFont(QFont('楷体', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('image/window_ui/logo.png'))
        about_img.setAlignment(Qt.AlignCenter)

        # label4.setText("<a href='https://oi.wiki/wiki/学习率的调整'>如何调整学习率</a>")
        label_super = QLabel()  # todo 作者信息
        label_super.setText("zxx")
        label_super.setFont(QFont('楷体', 16))
        label_super.setOpenExternalLinks(True)
        # label_super.setOpenExternalLinks(True)
        label_super.setAlignment(Qt.AlignRight)
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addStretch()
        about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)

        self.left_img.setAlignment(Qt.AlignCenter)
        self.addTab(img_detection_widget, '图片检测')
        self.addTab(about_widget, '关于')
        self.setTabIcon(0, QIcon('image/window_ui/logo.png'))
        self.setTabIcon(1, QIcon('image/window_ui/logo.png'))

    '''
    ***上传图片***
    '''

    def upload_img(self):
        # 选择图像文件进行读取
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            suffix = fileName.split(".")[-1]
            save_path = osp.join("image/tmp", "tmp_upload." + suffix)
            shutil.copy(fileName, save_path)
            # 应该调整一下图片的大小，然后统一放在一起
            im0 = cv2.imread(save_path)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("image/tmp/upload_show_result.jpg", im0)
            # self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))
            self.img2predict = fileName
            self.origin_shape = (im0.shape[1], im0.shape[0])
            self.left_img.setPixmap(QPixmap("image/tmp/upload_show_result.jpg"))
            # todo 上传图片之后右侧的图片重置，
            self.right_img.setPixmap(QPixmap("image/window_ui/right.png"))
            self.rrr.setText("等待识别")


    '''
    ***检测图片***
    '''
    # 写一个通用的内容，在主界面上，包含一些对水质的介绍。
    def detect_img(self):
        # model = self.model
        # output_size = self.output_size
        source = self.img2predict  # file/dir/URL/glob, 0 for webcam
        img = Image.open(source)
        img = self.valid_transforms(img)
        img = img.unsqueeze(0)
        img = img.to(device)
        output = self.model(img)
        label_id = torch.argmax(output).item()
        predict_name = self.classes_names[label_id]
        self.rrr.setText("当前识别结果为：{}".format(predict_name))

    # 关闭事件 询问用户是否退出
    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     '退出',
                                     "是否要退出程序？",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())