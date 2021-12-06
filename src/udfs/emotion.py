# coding=utf-8
# Copyright 2018-2020 EVA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List

from facenet_pytorch import MTCNN
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn.functional as F
from src.models.catalog.frame_info import FrameInfo
from src.models.catalog.properties import ColorSpace
from src.udfs.pytorch_abstract_udf import PytorchAbstractUDF
from skimage.transform import resize
from torch.autograd import Variable
from PIL import Image
from src.udfs.emotion.transforms import transforms as transforms
from src.udfs.emotion.vgg import VGG


class EmotionDetector(PytorchAbstractUDF):
    """
    Arguments:
        threshold (float): Threshold for classifier confidence score

    """

    @property
    def name(self) -> str:
        return "Emotion_Detector"

    def __init__(self):
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')

        self.model = MTCNN(keep_all=True, device=device)
        self.net = VGG('VGG19')
        checkpoint = torch.load('src/udfs/emotion/PrivateTest_model.t7')
        self.net.load_state_dict(checkpoint['net'])
        if device == torch.device('cuda:0'):
            self.net.cuda()
        self.net.eval()
        self.cut_size = 44
        self.transform_test = transforms.Compose([
            transforms.TenCrop(self.cut_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))])


    @property
    def input_format(self) -> FrameInfo:
        return FrameInfo(-1, -1, 3, ColorSpace.RGB)

    @property
    def labels(self) -> List[str]:
        return ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def _get_predictions(self, frames: Tensor) -> pd.DataFrame:
        """
        Performs predictions on input frames
        Arguments:
            frames (np.ndarray): Frames on which predictions need
            to be performed

        Returns:
            tuple containing predicted_classes (List[List[str]]),
            predicted_boxes (List[List[BoundingBox]]),
            predicted_scores (List[List[float]])

        """
        boxes, b_score = self.model.detect(frames.permute(2,3,1,0)[:,:,:,-1]*255)
        outcome = pd.DataFrame()
        N, C, row, col = frames.shape
        frame = frames.permute(2,3,1,0)[:,:,:,-1].cpu().numpy()*255

        rect = np.zeros((len(boxes), 2, 2))
        text = np.zeros(len(boxes)).astype(str)
        cnt = 0
        f_scores = []
        for box in boxes:
            wide_box = box + np.array([-10, -10, 10, 10])
            if wide_box[0] < 0 :
                wide_box[0] = 0
            if wide_box[1] < 0:
                wide_box[1] = 0
            if wide_box[2] >= col:
                wide_box[2] = col - 1
            if wide_box[3] >= row:
                wide_box[3] = row - 1
            
            x1, y1, x2, y2 = wide_box.astype(int)

            gray = np.dot(frame[y1:y2, x1:x2, :3], [0.299, 0.587, 0.114])
            gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)
            img = gray[:, :, np.newaxis]
            img = np.concatenate((img, img, img), axis=2)
            img = Image.fromarray(img)
            inputs = self.transform_test(img)

            ncrops, c, h, w = np.shape(inputs)
            inputs = inputs.view(-1, c, h, w)
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            device = torch.device('cpu')
            if device == torch.device('cuda:0'):
                inputs = inputs.cuda()
            
            inputs = Variable(inputs)
            outputs = self.net(inputs)
            
            outputs_avg = outputs.view(ncrops, -1).mean(0)
            _, predicted = torch.max(outputs_avg.data, 0)
            f_score = F.softmax(outputs_avg, dim = 0)
            f_score, _ = torch.max(f_score, 0)
            f_scores.append(float(f_score.cpu().detach().numpy()))
            emotion_text = self.labels[predicted.cpu().numpy()]
            
            

            text[cnt] = emotion_text
            rect[cnt, :, :] = [[x1, y1], [x2, y2]]
            cnt += 1

        outcome = outcome.append(
            {
                "labels": text,
                "scores": f_scores,
                "boxes": rect
            },
            ignore_index=True)

        return outcome
