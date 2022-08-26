import argparse
import numpy as np
import torch
import cv2
import logging
from preprocessing import Detector
from recognition.models import load_network

def draw_faces(infos, frame):
    for i in range(len(infos['bbs'])):
        cv2.rectangle(frame, (int(infos['bbs'][i][0]), int(infos['bbs'][i][1])), (int(infos['bbs'][i][2]), int(infos['bbs'][i][3])), color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        for j in range(len(infos['landmarks'][i])):
            cv2.circle(frame, (int(infos['landmarks'][i][j][0]), int(infos['landmarks'][i][j][1])), radius=4, color=(0, 255, 0), thickness=-1)


def webcam_recognition(preprocessing_method, model_name):
    vc = cv2.VideoCapture(1)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    ## Face Detector
    detector = Detector(preprocessing_method)

    ## Recognition Network
    net = load_network(model_name)
    net.eval()

    while True:
        rval, frame = vc.read()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_copy = frame_bgr.copy()
        
        infos, faces = detector.detect(frame_copy)

        try: 
            faces_transposed = np.transpose(faces, (0,3,1,2))
            faces_torch = torch.Tensor(faces_transposed)
            features = []
            for i in range(len(faces_torch)):
                face = faces_torch[i].unsqueeze(0)
                print(type(face))
                face = face.cuda()

                #res = [net(d.view(-1, d.shape[2], d.shape[3], d.shape[4])).data.cpu().numpy() for d in faces_torch]
                feature = net(face)
                print(face.shape)

                #feature = np.concatenate((res[0], res[1]), 1)
                features.append(feature)

            print(features)
            
        except Exception as e:
            logging.error('Error in detection(No face detected): ' + str(e))

        
        try:        
            cv2.imwrite('res/img_teste.jpeg', faces[0])
            draw_faces(infos, frame)
        except Exception as e:
            logging.error('Error in detection(No face detected): ' + str(e))

        

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vc.release()
    cv2.destroyWindow("Video")

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessing_method',
                       type=str,
                       default='mtcnn')
    parser.add_argument('--model_name',
                       type=str,
                       default='arcface')

    args = parser.parse_args()
    print(args)

    webcam_recognition(args.preprocessing_method, args.model_name)