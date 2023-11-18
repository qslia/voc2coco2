import os
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pathlib import Path
import shutil


class CvtVoc2Yolo:
    def __init__(self, img_dir, xml_dir, yolo_dir):
        self.img_dir = img_dir
        self.xml_dir = xml_dir
        if not os.path.exists(yolo_dir):
            os.mkdir(yolo_dir)
        self.yolo_dir = yolo_dir
        self.boxes = []
        self.labels = []
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                        'sofa', 'train', 'tvmonitor']

    def __len__(self):
        return len(os.listdir(self.xml_dir))

    def _cvt(self, xml):
        xml_path = os.path.join(os.path.abspath('.'), xml_dir + '/' + xml)

        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        size_ = [width, height]

        img_box = []
        for object in root.iter('object'):
            difficult = int(object.find('difficult').text == '1')
            cls_name = object.find('name').text.lower().strip()
            bbox = object.find('bndbox')
            xmin = int(bbox.find('xmin').text) - 1
            ymin = int(bbox.find('ymin').text) - 1
            xmax = int(bbox.find('xmax').text) - 1
            ymax = int(bbox.find('ymax').text) - 1
            img_box.append([cls_name, xmin, ymin, xmax, ymax])

        return size_, img_box, xml_path

    def save_txt_file(self, size_, img_box, xml_path):
        save_file_name = os.path.abspath(yolo_dir) + '/' + os.path.basename(xml_path).split('.')[0] + '.txt'
        with open(save_file_name, 'a+') as f:
            for box in img_box:
                cls_num = int(self.classes.index(box[0]))
                new_box = self.xyxy2xywh(size_, box[1:])
                f.write(f"{cls_num} {new_box[0]} {new_box[1]} {new_box[2]} {new_box[3]}\n")

    def xyxy2xywh(self, size, box):
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        dw = np.float32(1. / int(size[0]))
        dh = np.float32(1. / int(size[1]))

        w = x2 - x1
        h = y2 - y1
        x = x1 + (w / 2)
        y = y1 + (h / 2)

        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return [x, y, w, h]

    def cvt(self):
        for xml in tqdm(os.listdir(self.xml_dir)):
            size_, img_box, xml_path = self._cvt(xml)
            self.save_txt_file(size_, img_box, xml_path)
        print('done')
        print(f'yolo format .txt saved at {os.path.abspath(yolo_dir)}')

    def split_dataset(self):
        if not os.path.exists('Dataset/images'):
            os.makedirs('Dataset/images')
        if not os.path.exists('Dataset/labels'):
            os.makedirs('Dataset/labels')
        img_abs_path = os.path.realpath(self.img_dir)
        images = os.listdir(self.img_dir)
        self.train_images, test_images = train_test_split(images, test_size=0.2, random_state=11)
        self.val_images, self.test_images = train_test_split(test_images, test_size=0.5, random_state=11)
        print(f'total = {len(images)} images\n'
              f'train = {len(self.train_images)} images\n'
              f'test  = {len(self.test_images)} images\n'
              f'val  = {len(self.val_images)} images\n')

    def _copy_file(self, images, mode):
        img_target = Path('Dataset/images/' + mode)
        if not img_target.exists():
            os.makedirs(img_target)
        label_target = Path('Dataset/labels/' + mode)
        if not label_target.exists():
            os.makedirs(label_target)

        for image in images:
            image_name = image.split('.')[0]
            img_dir_abs_path = os.path.realpath(self.img_dir)
            img_src = os.path.join(img_dir_abs_path, image)
            label_src_abs_path = os.path.realpath(yolo_dir)
            label_src = os.path.join(label_src_abs_path, image.split('.')[0] + '.txt')
            shutil.copy(img_src, img_target)
            shutil.copy(label_src, label_target)

    def copy_file(self):
        self.split_dataset()
        file_list = [self.train_images, self.test_images, self.val_images]
        mode = ['train', 'test', 'val']
        for images, _mode in zip(file_list, mode):
            self._copy_file(images, _mode)


if __name__ == '__main__':
    img_dir = 'JPEGImages'
    xml_dir = 'Annotations'
    yolo_dir = 'yolo_dir'
    a = CvtVoc2Yolo(img_dir, xml_dir, yolo_dir)
    print(len(a))
    a.copy_file()
