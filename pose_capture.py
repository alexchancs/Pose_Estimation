#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# MIT License
#
# Copyright (c) 2019, 2020 MACNICA Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#


import os
import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import torchvision.transforms as transforms
from trt_pose.parse_objects import ParseObjects
import re
import logging
from collections import deque
import numpy as np
import time
import cv2

class PoseCaptureError(Exception):
    pass


class PoseCaptureDescError(PoseCaptureError):
    pass


class PoseCaptureModelError(PoseCaptureError):
    pass

class PoseCaptureModel:

    def __init__(self, modelFile, taskDescFile):

        # Load the task description
        try:
            with open(taskDescFile, 'r') as f:
                human_pose = json.load(f)
        except OSError:
            raise PoseCaptureDescError
        self.topology = trt_pose.coco.coco_category_to_topology(human_pose)
        self.part_list = human_pose['keypoints']
        self.link_list = human_pose['skeleton']

        # Load the base model
        fbase = os.path.basename(modelFile)
        func, self.inWidth, self.inHeight = PoseCaptureModel.getModelFuncName(fbase)
        if func is None:
            logging.fatal('Invalid model name: %s' % fbase)
            logging.fatal('Model name should be (.+_.+_att)_(\\d+)x(\\d+)_')
            raise PoseCaptureModelError('Invalid model name: %s' % (fbase))
        if not hasattr(trt_pose.models, func):
            logging.fatal('Could not find base model function: %s' % (func))
            raise PoseCaptureModelError(
                'Could not find base model function: %s' % (func))
        func = 'trt_pose.models.' + func
        trtFile = os.path.splitext(fbase)[0] + '_trt.pth'
        logging.info('Loading base model from %s' % (func))
        model = eval(func)(len(self.part_list), 2 * len(self.link_list)).cuda().eval()

        if os.path.exists(trtFile):
            logging.info('Loading model from TensorRT plan file ...')
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trtFile))
        else:
            logging.info('Optimizing model for TensorRT ...')
            model.load_state_dict(torch.load(modelFile))
            data = torch.zeros((1, 3, self.inHeight, self.inWidth)).cuda()
            model_trt = torch2trt.torch2trt(
                model, [data], fp16_mode=True, max_workspace_size=1 << 25)
            torch.save(model_trt.state_dict(), trtFile)

        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.parse_objects = ParseObjects(self.topology)
        self.model_trt = model_trt

    def process(self, image):
        t0 = time.time()
        #resized_image = cv2.resize(image, (self.inWidth, self.inHeight), interpolation=cv2.INTER_NEAREST)
        # Resize GPU
        srcGpu = cv2.cuda_GpuMat()
        dstGpu = cv2.cuda_GpuMat()
        srcGpu.upload(image)
        dstGpu = cv2.cuda.resize(srcGpu, (self.inWidth, self.inHeight), interpolation=cv2.INTER_NEAREST)
        resized_image = dstGpu.download()


        t1 = time.time()
        tensor_img = transforms.ToTensor()(resized_image).cuda()
        input_img = self.normalize(tensor_img)
        t2 = time.time()
        t_cmap, t_paf = self.model_trt(input_img[None, ...])
        cmap, paf = t_cmap.detach().cpu(), t_paf.detach().cpu()
        t3 = time.time()
        counts, objects, peaks = self.parse_objects(cmap, paf)
        t4 = time.time()
        humans = self.decode_skeleton_pt(image, counts, objects, peaks)
        t5 = time.time()

        # print('Resize: ', t1 - t0)
        # print('Normalize: ', t2 - t1)
        # print('Inference: ', t3 - t2)
        # print('Parse: ', t4 - t3)
        # print('Decode: ', t5 - t4)
        # print('Overall: ', t5 - t0)

        return humans

    def decode_skeleton_pt(self, image, object_counts, objects, normalized_peaks):
        topology = self.topology
        shape = image.shape
        height, width = shape[0], shape[1]

        K = topology.shape[0]
        count = int(object_counts[0])

        detected_human = deque()

        for i in range(count):
            body = SkeletonBody(self.part_list, self.link_list)
            detected_human.append(body)
            obj = objects[0][i]
            C = obj.shape[0]
            for j in range(C):
                k = int(obj[j])
                if k >= 0:
                    peak = normalized_peaks[0][j][k]
                    x = int(peak[1] * width)
                    y = int(peak[0] * height)

                    body.position(self.part_list[j], (x, y))

            for k in range(K):
                c_a = topology[k][2]
                c_b = topology[k][3]
                if obj[c_a] >= 0 and obj[c_b] >= 0:
                    peak0 = normalized_peaks[0][c_a][obj[c_a]]
                    peak1 = normalized_peaks[0][c_b][obj[c_b]]
                    x0 = int(peak0[1] * width)
                    y0 = int(peak0[0] * height)
                    x1 = int(peak1[1] * width)
                    y1 = int(peak1[0] * height)

                    body.lines.append(((x0, y0), (x1, y1)))
        return detected_human

    @staticmethod
    def getModelFuncName(model_file):
        result = re.search('(.+_.+_att)_(\d+)x(\d+)_', model_file)
        if result is None:
            return None, None, None
        else:
            return result[1], int(result[2]), int(result[3])
            
    @staticmethod
    def add_parse_argument(parser):
        parser.add_argument('--model',
                            type=str,
                            default='resnet18_baseline_att_224x224_A_epoch_249.pth',
                            metavar='MODEL',
                            help='Model weight file')
        parser.add_argument('--task',
                            type=str,
                            default='human_pose.json',
                            metavar='TASK_DESC',
                            help='Task description file')
        return parser

class SkeletonBody:
    def __init__(self, point_list, skeleton_list):
        self.skeletons = skeleton_list
        self.body_part = {}
        self.lines = deque()
        for point in point_list:
            self.body_part[point] = (-1,-1)

        self.body_part['center_hip'] = (-1,-1)

    def position(self, point: str, pos=(None, None)):
        if self.body_part.get(point) is None:
            return None
        if None not in pos:
            self.body_part[point] = pos
            if 'hip' in point and 'center' not in point:
                if self.body_part['left_hip'] >= (0,0) and self.body_part['right_hip'] >= (0,0):
                    hips = (self.body_part['left_hip'], self.body_part['right_hip'])
                    self.body_part['center_hip'] = tuple(np.mean(hips, axis=0, dtype=np.int).tolist())
        return self.body_part.get(point)




