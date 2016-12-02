# -*- coding: utf-8 -*-

import chainer
import argparse
import numpy as np
import cv2
from chainer import cuda
from chainer import serializers
from chainer import Variable
from chainer import optimizers
import chainer.links as L
import chainer.functions as F


class VGGNet(chainer.Chain):

    def __init__(self):
        super(VGGNet, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),

            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            fc6=L.Linear(25088, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 1000)
        )
        self.train = False

    def __call__(self, x, t):

        self.x = x
        self.h = F.relu(self.conv1_1(x))
        self.h1= F.relu(self.conv1_2(self.h))
        self.h = F.max_pooling_2d(self.h1, 2, stride=2)

        self.h = F.relu(self.conv2_1(self.h))
        self.h2= F.relu(self.conv2_2(self.h))
        self.h = F.max_pooling_2d(self.h2, 2, stride=2)

        self.h = F.relu(self.conv3_1(self.h))
        self.h = F.relu(self.conv3_2(self.h))
        self.h3= F.relu(self.conv3_3(self.h))
        self.h = F.max_pooling_2d(self.h3, 2, stride=2)

        self.h = F.relu(self.conv4_1(self.h))
        self.h = F.relu(self.conv4_2(self.h))
        self.h4= F.relu(self.conv4_3(self.h))
        self.h = F.max_pooling_2d(self.h4, 2, stride=2)

        self.h = F.relu(self.conv5_1(self.h))
        self.h = F.relu(self.conv5_2(self.h))
        self.h5= F.relu(self.conv5_3(self.h))
        self.h = F.max_pooling_2d(self.h, 2, stride=2)

        self.h = F.dropout(F.relu(self.fc6(self.h)), train=self.train, ratio=0.5)
        self.h = F.dropout(F.relu(self.fc7(self.h)), train=self.train, ratio=0.5)
        self.h = self.fc8(self.h)

        if self.train:
            self.lossx = F.softmax_cross_entropy(self.h, t)
            self.acc = F.accuracy(self.h, t)
            return self.lossx
        else:
            self.pred = F.softmax(self.h)
            return self.pred


class COLORNet(chainer.Chain):

    def __init__(self):
        super(COLORNet, self).__init__(
            fx0=L.Linear(1000, 512),
            fx1=L.Linear(512, 256),

            convf_0=L.Convolution2D(512, 256, 1, stride=1, pad=0),

            bn0 = L.BatchNormalization(512),
            conv5_1=L.Convolution2D(512, 512, 1, stride=1, pad=0),
            deconv5_1 = L.Deconvolution2D(512, 512, 2, stride=2, pad=0),

            conv4_1=L.Convolution2D(512, 256, 1, stride=1, pad=0),
            bn1 = L.BatchNormalization(512),
            deconv4_1 = L.Deconvolution2D(256, 256, 2, stride=2, pad=0),
            bn2 = L.BatchNormalization(256),

            conv4_2=L.Convolution2D(256, 128, 3, stride=1, pad=1),
            bn3 = L.BatchNormalization(128),

            deconv3_1 = L.Deconvolution2D(128, 128, 2, stride=2, pad=0),
            bn4 = L.BatchNormalization(128),

            conv3_1=L.Convolution2D(128, 64, 3, stride=1, pad=1),
            bn5 = L.BatchNormalization(64),

            deconv2_1 = L.Deconvolution2D(64, 64, 2, stride=2, pad=0),
            bn6 = L.BatchNormalization(64),

            conv2_1=L.Convolution2D(64, 3, 3, stride=1, pad=1),
            bn7 = L.BatchNormalization(3),
            bn8 = L.BatchNormalization(3),

            conv1_1=L.Convolution2D(3, 3, 3, stride=1, pad=1),
            bn9 = L.BatchNormalization(3),

            conv0_5=L.Convolution2D(3, 2, 3, stride=1, pad=1),
        )

    def __call__(self, t, sw):

        hf0 = self.fx0(vgg.h)
        hf1 = self.fx1(hf0)

        fusionG = xp.tile(hf1.data,(14,14,1))
        fusionG = fusionG.transpose(2,0,1)
        fusionG = fusionG[np.newaxis,:,:,:]

        fusionL = self.convf_0(vgg.h5)
        fusionL = fusionL.data
        fusion  = xp.concatenate([fusionG, fusionL], axis=1)

        h0 = F.relu(self.conv5_1(self.bn0(Variable(fusion))))
        h0 = self.deconv5_1(h0)

        h1 = self.bn1(vgg.h4)
        h2 = h0 + h1
        h2 = self.conv4_1(h2)
        h2 = F.relu(self.bn2(h2))

        h2 = self.deconv4_1(h2)
        h3 = self.bn2(vgg.h3)
        h4 = h2 + h3

        h4 = self.conv4_2(h4)
        h4 = F.relu(self.bn3(h4))
        h4 = self.deconv3_1(h4)
        h5 = self.bn4(vgg.h2)
        h5 = h4 + h5

        h5 = self.conv3_1(h5)
        h5 = F.relu(self.bn5(h5))
        h5 = self.deconv2_1(h5)
        h6 = self.bn6(vgg.h1)
        h6 = h5 + h6

        h6 = self.conv2_1(h6)
        h6 = F.relu(self.bn7(h6))
        h7 = self.bn8(vgg.x)
        h8 = h6 + h7

        h8 = self.conv1_1(h8)
        h8 = F.relu(self.bn9(h8))

        h8 = self.conv0_5(h8)

        zx81 = F.split_axis(h8,1,0)
        zx82 = F.split_axis(zx81,2,1)

        if sw == 1:
            t1 = F.reshape(t,(1, 224*224))
            x = F.reshape(zx82[0],(1, 224*224))
            self.loss = F.mean_squared_error(x, t1)
            return self.loss

        elif sw == 2:
            t1 = F.reshape(t,(1, 224*224))
            x = F.reshape(zx82[1],(1, 224*224))
            self.loss = F.mean_squared_error(x, t1)
            return self.loss

        else:
            return h8


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("inputMovie", type=file)
    parser.add_argument("outputMovie")
    parser.add_argument("--parm", default=0.0002, type=float)
    parser.add_argument("--mon", const=1, nargs="?")
    parser.add_argument("--epoch", default=1, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--file", default="mahalita10", type=str)
    parser.add_argument("--load", default=0, type=int)
    parser.add_argument("--savestep", default=10000, type=int)
    parser.add_argument("--test", const=1, nargs="?")
    parser.add_argument("--log", default=100, type=int)
    parser.add_argument("--fast", const=1, nargs="?")

    args = parser.parse_args()
    inputMovie = args.inputMovie.name
    outputMovie = args.outputMovie

    cap = cv2.VideoCapture(inputMovie)
    heightInputMovie = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    widthInputMovie = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    allFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = "XVID"
    out = cv2.VideoWriter(outputMovie, cv2.VideoWriter_fourcc(*fourcc), FPS, (widthInputMovie,heightInputMovie))

    vgg_mean = np.array([123.68, 116.779, 103.939 ])

    vgg = VGGNet()
    colornet = COLORNet()

    optimizer = optimizers.SGD(args.parm)
    optimizer.setup(colornet)

    serializers.load_hdf5('VGG.model', vgg)

    if args.load :

         serializers.load_hdf5(args.file + "_%d.model" % args.load, colornet)
         serializers.load_hdf5(args.file + "_%d.state" % args.load, optimizer)
         print ("Load model: " +args.file + "_%d.model" % args.load)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        vgg.to_gpu()
        colornet.to_gpu()

    xp = cuda.cupy if args.gpu >= 0 else np

    iteratecnt = args.load
    inputd = np.zeros((1, 3, 224, 224), dtype=np.float32)

    for vplaycnt in range (args.epoch):
        for frameNo in range(1, allFrames+1):

            ret, frame = cap.read()

            if ret == True:

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
                frame_rgb_f = (frame_rgb / 255)
                frame_rgb -= vgg_mean

                frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                frame_gray_vgg = cv2.resize(frame_gray, (224, 224))

                inputd[0,0,:,:] = frame_gray_vgg
                inputd[0,1,:,:] = frame_gray_vgg
                inputd[0,2,:,:] = frame_gray_vgg

                pred = vgg(Variable(xp.asarray(inputd, dtype=np.float32)), None)

                frame_lab = cv2.cvtColor(frame_rgb_f, cv2.COLOR_RGB2Lab)
                frame_lab_l = frame_lab[:,:,0]
                frame_lab_vgg = cv2.resize(frame_lab, (224, 224))
	
                labd = xp.asarray(frame_lab_vgg, dtype=xp.float32)

                if not args.test :

                    colornet.zerograds()
                    loss1 = colornet(Variable(labd[:,:,1]),1)
                    loss1.backward()
                    optimizer.update()

                    colornet.zerograds()
                    loss2 = colornet(Variable(labd[:,:,2]),2)
                    loss2.backward()
                    optimizer.update()

                    if iteratecnt % args.log == 0:
                        print ("{0},{1},{2}".format(iteratecnt, loss1.data, loss2.data))

                if not args.fast :

                    pred = colornet(None, 3)

                    pred_lab_a = xp.array(pred.data[0,0,:,:])
                    if args.gpu >= 0:
                        pred_lab_a = cuda.to_cpu(pred_lab_a)
                    pred_lab_a = cv2.resize(pred_lab_a,(widthInputMovie, heightInputMovie))

                    pred_lab_b = xp.array(pred.data[0,1,:,:])
                    if args.gpu >= 0:
                        pred_lab_b = cuda.to_cpu(pred_lab_b)
                    pred_lab_b = cv2.resize(pred_lab_b,(widthInputMovie, heightInputMovie))

                    frame_lab_out = np.concatenate((frame_lab_l[:,:,np.newaxis],pred_lab_a[:,:,np.newaxis],pred_lab_b[:,:,np.newaxis]),axis=2)

                    frame_rgb_out = cv2.cvtColor(frame_lab_out, cv2.COLOR_Lab2RGB)

                    frame_rgb_out = (frame_rgb_out * 255).astype(np.uint8)
                    cvFrame = cv2.cvtColor(frame_rgb_out, cv2.COLOR_RGB2BGR)
                    out.write(cvFrame)

                    if args.mon:
                        cv2.imshow('frame', cvFrame)
                        cv2.waitKey(1)

                iteratecnt += 1

            if iteratecnt % args.savestep == 0 and not args.test:
                serializers.save_hdf5(args.file + "_%d.model" % iteratecnt, colornet)
                serializers.save_hdf5(args.file + "_%d.state" % iteratecnt, optimizer)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if not args.test:
        serializers.save_hdf5(args.file + "_%d.model" % iteratecnt, colornet)
        serializers.save_hdf5(args.file + "_%d.state" % iteratecnt, optimizer)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

