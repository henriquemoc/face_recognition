import os
import argparse
import pickle

import numpy as np
import cv2
from PIL import Image
from torchvision import transforms as T
import torch
import matplotlib.pyplot as plt

from preprocessing import Detector
from recognition.models import load_network


class GenericDataloader:

       def __init__(self, dataset_directory, detector):

              self.dataset_directory = dataset_directory
              self.detector = detector

              self.file_list = []
              for filename in os.listdir(dataset_directory):
                     self.file_list.append(os.path.join(dataset_directory, filename))
              print(self.file_list)

       def __getitem__(self, index):
              
              img = cv2.imread(self.file_list[index])
              
              plt.imshow(img)
              #Transformação para RGB
              img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

              infos, faces = self.detector.detect(img)

              faces_transposed = np.transpose(faces, (0,3,1,2))
              img = faces_transposed[0]

              #Aplicando transformações para se adequar ao modelo para o treinamento
              img_pil = Image.fromarray(img)

              normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                   std=[0.5, 0.5, 0.5])
              img_transforms = T.Compose([T.ToTensor(), normalize])

              img = img_transforms(img_pil)

              return img


       def __len__(self):
              return len(self.file_list)

def get_features_from_dataset(feature_file):

       features = None

       assert os.path.isdir(os.path.dirname(feature_file)) or os.path.dirname(feature_file)=="", \
              "Feature file directory must exist"
       
       if feature_file is not None and os.path.isfile(feature_file):
              with open(feature_file, 'rb') as handle:
                     features = pickle.load(handle)
       
       return features

def extract_features_from_dataset(preprocessing_method, model_name, dataset_directory):

       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

       features=None

       assert os.path.isdir(dataset_directory) or os.path.dirname(dataset_directory)=="", \
              "Dataset directory must exist"
       
       ## Face Detector
       detector = Detector(preprocessing_method)

       ## Recognition Network
       net = load_network(model_name)
       net.eval()

       dataset = GenericDataloader(dataset_directory, detector)
       dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

       for i, data in enumerate(dataloader, 0):

              features = []
              for i in range(len(data)):
                     face = data[i].unsqueeze(0)
                     print(type(face))
                     face = face.cuda()

                     #res = [net(d.view(-1, d.shape[2], d.shape[3], d.shape[4])).data.cpu().numpy() for d in faces_torch]
                     feature = net(face)
                     print(face.shape)

                     #feature = np.concatenate((res[0], res[1]), 1)
                     features.append(feature)
                            
              print(features)


if __name__ == "__main__":
       parser = argparse.ArgumentParser()
       parser.add_argument('--preprocessing_method',
                            type=str,
                            default='mtcnn')
       parser.add_argument('--model_name',
                            type=str,
                            default='arcface')
       parser.add_argument('--dataset_directory',
                            type=str,
                            default='dataset')

       args = parser.parse_args()
       print(args)

       extract_features_from_dataset(args.preprocessing_method, args.model_name, args.dataset_directory)