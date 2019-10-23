import os
import glob
import shutil
import random
import time
from multiprocessing import Process, Queue, Manager
import dlib
import cv2
import numpy as np



def rect_to_bb(rect): # 获得人脸矩形的坐标信息
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def load_feature(path):
    data=np.load(path)
    return data['feat']

class make_Dataset():
    def __init__(self,id,path):
        self.id=id
        self.path=path

    def collect_images(self,num):
        cam = cv2.VideoCapture(0)
        cam.set(3, 640)  # set video width
        cam.set(4, 480)  # set video height

        path = os.path.join(self.path, self.id)
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path)

        detector = dlib.get_frontal_face_detector()
        print('开始采集！')
        count = 0
        total = num
        while (True):
            ret, img = cam.read()
            if ret:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 1)

                # 选择最大面积的人脸作为主要人脸
                face = None
                if len(rects) >= 1:
                    max_rect = None
                    max_area = 0
                    for (i, rect) in enumerate(rects):
                        (x, y, w, h) = rect_to_bb(rect)
                        area = w * h
                        if area > max_area:
                            max_rect = rect
                            max_area = area
                    face = max_rect

                if face is not None:
                    count += 1
                    cv2.imwrite(os.path.join(path, str(face_id) + '_' + str(count) + ".jpg"),
                                img)

                k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
                if k == 27:
                    break
                elif count >= total:  # 采集total张后结束采集
                    break

    def make_feature(self,file_path):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
        face_model=dlib.face_recognition_model_v1("./dlib_face_recognition_resnet_model_v1.dat")

        img_path=os.path.join(self.path,self.id)
        feature=[]
        for sub_path in glob.glob(os.path.join(img_path,'*.jpg')):
            img=cv2.imread(sub_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)
            print(len(rects))

            # 选择最大面积的人脸作为主要人脸
            face = None
            if len(rects) >= 1:
                max_rect = None
                max_area = 0
                for (i, rect) in enumerate(rects):
                    (x, y, w, h) = rect_to_bb(rect)
                    area = w * h
                    if area > max_area:
                        max_rect = rect
                        max_area = area
                face = max_rect

            if face is not None:
                shape=predictor(img,face)
                face_descriptor=face_model.compute_face_descriptor(img,shape)
                feature.append(list(face_descriptor))

        feature=np.array(feature)
        print(feature.shape)
        np.savez(os.path.join(file_path,str(self.id)+'.npz'),feat=feature)


def cam_get_image(queen_image:Queue,num,path):
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height

    detector = dlib.get_frontal_face_detector()
    count = 0
    total = num
    while True:
        print('pr1')
        ret, img = cam.read()
        if ret:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)

            # 选择最大面积的人脸作为主要人脸
            face = None
            if len(rects) >= 1:
                max_rect = None
                max_area = 0
                for (i, rect) in enumerate(rects):
                    (x, y, w, h) = rect_to_bb(rect)
                    area = w * h
                    if area > max_area:
                        max_rect = rect
                        max_area = area
                face = max_rect

            if face is not None:
                count+=1
                cv2.imwrite(os.path.join(path,str(count)+'.jpg'),img)
                queen_image.put([count,img])
                print('pr1:',count)

            k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                break
            elif count >= total:  # 采集total张后结束采集
                break

def cal_image(queen_image:Queue,queen_feature:Queue,num):
    while True:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
        face_model = dlib.face_recognition_model_v1("./dlib_face_recognition_resnet_model_v1.dat")

        count=0
        if not queen_image.empty():
            count+=1
            print('pr2')
            count,img = queen_image.get(True)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)

            # 选择最大面积的人脸作为主要人脸
            face = None
            if len(rects) >= 1:
                max_rect = None
                max_area = 0
                for (i, rect) in enumerate(rects):
                    (x, y, w, h) = rect_to_bb(rect)
                    area = w * h
                    if area > max_area:
                        max_rect = rect
                        max_area = area
                face = max_rect

            if face is not None:
                shape = predictor(img, face)
                face_descriptor = face_model.compute_face_descriptor(img, shape)
                feature = np.array(face_descriptor).reshape(1,-1)
                queen_feature.put([feature])

            if count>=num:
                break

def predict_image(queen_feature:Queue,feature_path,result_list):
    while True:
        if not queen_feature.empty():
            print('pr3')
            feature=queen_feature.get(True)[0]
            name_dist_list=[]
            for feat in os.listdir(feature_path):
                name = feat.split('.')[0]
                feat_path = os.path.join(feature_path, feat)
                feat_mat = load_feature(feat_path)
                dist = np.linalg.norm(feat_mat - feature, axis=1)
                dist_sort = sorted(list(dist.flatten()))
                name_dist_list.append([dist_sort[0],name])

            name_dist_sort=sorted(name_dist_list,key=lambda x:x[0])
            print(name_dist_sort[0][0])
            if name_dist_sort[0][0]<0.4:
                result_list.append(name_dist_sort[0][1])
            else:
                result_list.append('unknown')


if __name__ == '__main__':
    # face_id = input('\n enter user id end press <return> ==>  ')
    # dataset_path = './dataset'
    # dataset=make_Dataset(face_id,dataset_path)
    # dataset.collect_images(100)
    # dataset.make_feature('./feature')
    start = time.time()
    test_path='./output'
    feature_path='./feature'
    queen_size=15
    num=5
    queen_image=Queue(queen_size)
    queen_feature=Queue(queen_size)
    with Manager() as mg:
        #使用Manager()后建立的list可以获取多个进程的数据，直接使用list最终进程结束后输出为空
        result_list = mg.list([])
        pr1=Process(target=cam_get_image,args=(queen_image,num,test_path,))
        pr2=Process(target=cal_image,args=(queen_image,queen_feature,num,))
        pr3=Process(target=predict_image,args=(queen_feature,feature_path,result_list))

        pr1.start()
        pr2.start()
        pr3.start()

        pr1.join()
        pr2.join()
        pr3.terminate()

        print(result_list)

    finish=time.time()
    print(finish-start)