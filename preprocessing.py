import torch
from detection.mtcnn import MTCNN
from detection.utils_mtcnn import *
from retinaface import RetinaFace
from retinaface.RetinaFace import build_model, detect_faces, extract_faces
import logging


class Detector:
    def __init__(self, preprocessing_method = 'mtcnn'):
        self.preprocessing_method = preprocessing_method
        self.detector = None
        if self.preprocessing_method == 'mtcnn':
            self.detector = MTCNN(keep_all=True, selection_method="largest", post_process=False, image_size=(112,112),
                           device=torch.device('cuda'))
        elif self.preprocessing_method == 'retinaface':
            self.detector = build_model()
        self.infos_detected = {'bbs': None, 'landmarks': None, 'scores': None}
        self.faces_detected = []
    
    def detect(self, frame):
        faces = None
        if self.preprocessing_method =="mtcnn":
            try:
                bounding_boxes, probs, landmarks = self.detector.detect(frame, landmarks=True)
                self.infos_detected['bbs'] = bounding_boxes
                self.infos_detected['landmarks'] = landmarks
                self.infos_detected['scores'] = probs
                
                #landmarks = np.concatenate((landmarks[:, :, 0], landmarks[:, :, 1]), axis=1)
                
                faces = self.detector.extract(frame, bounding_boxes, None)

            except Exception as e:
                logging.error('Error in MTCNN: ' + str(e))

        elif self.preprocessing_method == "retinaface":
            try:
                thresh = 0.8
                resp, faces = extract_faces(frame, threshold=thresh, model=self.detector)
                #faces, landmarks = retinaface2.detect(frame_copy, thresh)
                bbs = [indiv_face['facial_area'] for indiv_face in resp.values()]
                self.infos_detected['bbs'] = bbs
                landmarks = [list(indiv_face['landmarks'].values()) for indiv_face in resp.values()]
                self.infos_detected['landmarks'] = landmarks
                scores = [indiv_face['score'] for indiv_face in resp.values()]
                self.infos_detected['scores'] = scores
                
            except Exception as e:
                logging.error('Error in RetinaFace: ' + str(e))
        
        return (self.infos_detected, faces)