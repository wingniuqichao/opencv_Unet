
import cv2
import numpy as np
import time
import argparse

cv2.namedWindow("原始", cv2.WINDOW_NORMAL)
cv2.namedWindow("黑白", cv2.WINDOW_NORMAL)
cv2.namedWindow("标注", cv2.WINDOW_NORMAL)
cv2.namedWindow("扣取", cv2.WINDOW_NORMAL)
# 读取模型
net = cv2.dnn.readNetFromCaffe("unet.prototxt", 'unet.caffemodel')
def predict(frame):
    frame = cv2.resize(frame, (256, 256))

    cv2.imshow("原始", frame)

    inputBlob = cv2.dnn.blobFromImage(frame, 1/255.0, (256, 256), (127.5, 127.5, 127.5), False)

    # 预测
    net.setInput(inputBlob, 'data')
    pred = net.forward("predict")
    
    # 获取结果
    pred = pred[0,1,:,:]
    
    pred[pred>0.5] = 255
    pred[pred<=0.5] = 0
    pred = np.array(pred, dtype=np.uint8)
    # 将人像扣取出来
    frame_person = frame.copy()
    frame_person[pred==0] = [255, 255, 255]
    # 将人像用红色标注
    frame[:,:,2][pred==255] = 255
    
    t2 = time.time()
    
    cv2.imshow('黑白', pred)
    cv2.imshow('标注', frame)
    cv2.imshow('扣取', frame_person)


def main():
    
    if args.video:
        cap = cv2.VideoCapture(args.video)
        while(1):
            t1 = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            predict(frame)
            cv2.waitKey(1)
    elif args.image:
        img = cv2.imread(args.image, 1)
        predict(img)
        cv2.waitKey(0)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='command for predict unet model')
    parse.add_argument('--image', type=str, default=None, help='the image to predict')
    parse.add_argument('--video', type=str, default='test/cxk.mp4', help='the video to predict')
    args = parse.parse_args()
    main()