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
from src.models.catalog.frame_info import FrameInfo
from src.models.catalog.properties import ColorSpace
from src.udfs.pytorch_abstract_udf import PytorchAbstractUDF


class FaceDetector(PytorchAbstractUDF):
    """
    Arguments:
        threshold (float): Threshold for classifier confidence score

    """

    @property
    def name(self) -> str:
        return "facenet_pytorch"

    def __init__(self):
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = MTCNN(keep_all=True, device=device)

    @property
    def input_format(self) -> FrameInfo:
        return FrameInfo(-1, -1, 3, ColorSpace.RGB)

    @property
    def labels(self) -> List[str]:
        return []

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

        boxes, scores = self.model.detect(frames.permute(2,3,1,0)[:,:,:,-1]*255)
        outcome = pd.DataFrame()
        N, C, row, col = frames.shape

        arr = np.zeros((len(boxes), 2, 2))
        cnt = 0
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
            
            x1, y1, x2, y2 = wide_box
            arr[cnt, :, :] = [[x1, y1], [x2, y2]]
            cnt += 1


        outcome = outcome.append(
            {
                "labels": 'face',
                "scores": scores,
                "boxes": arr
            },
            ignore_index=True)

        return outcome
