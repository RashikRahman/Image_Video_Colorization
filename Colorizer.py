import numpy as np
import cv2
import time
from os.path import splitext, basename, join

from numpy.core.records import format_parser

class Colorizer:
    def __init__(self, height=400, width=400, use_cuda=False):
        self.height, self.width = height, width
        self.colorModel = cv2.dnn.readNetFromCaffe('Model/colorization_deploy_v2.prototxt', 
                                                    caffeModel = 'Model/colorization_release_v2.caffemodel')

        if use_cuda:
            self.colorModel.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.colorModel.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        clusterCenters = np.load('Model/pts_in_hull.npy')
        clusterCenters = clusterCenters.transpose().reshape(2,313,1,1)

        self.colorModel.getLayer(self.colorModel.getLayerId('class8_ab')).blobs = [clusterCenters.astype(np.float32)]
        self.colorModel.getLayer(self.colorModel.getLayerId('conv8_313_rh')).blobs = [np.full([1,313], 2.606, dtype = np.float32)]
    
    def processImg(self, imgPath):
        self.img = cv2.imread(imgPath)
        self.img = cv2.resize(self.img, (self.width, self.height))
        self.processFrame()
        cv2.imwrite(join('Output', basename(imgPath)), self.ImgFinal)
        cv2.imshow('output', self.ImgFinal)
        cv2.waitKey(0)
        
    
    def processVideo(self, videoPath):
        cap = cv2.VideoCapture(videoPath)
        if cap.isOpened()==False:
            print('Error')
            return
        
        sucess = True
        prevFrameTime = 0
        nextFrameTime = 0
        out = cv2.VideoWriter(join('Output', splitext(basename(videoPath))[0]+'.avi'),
                                                cv2.VideoWriter_fourcc(*'MJPG'), 
                                                cap.get(cv2.CAP_PROP_FPS),
                                                (self.width*2, self.height))
        
        while sucess:
            sucess, self.img = cap.read()
            if self.img is None:
                break

            self.img = cv2.resize(self.img, (self.width, self.height))
            self.processFrame()
            out.write(self.ImgFinal)
            nextFrameTime = time.time()
            fps = 1/(nextFrameTime - prevFrameTime)
            prevFrameTime = nextFrameTime
            fps = 'FPS: '+str(int(fps))

            cv2.putText(self.ImgFinal, fps, (5,25), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow('output', self.ImgFinal)
            key  = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()



    def processFrame(self):
        imgNormalized = (self.img[:,:,[2,1,0]] * 1.0/255)
        imgNormalized = imgNormalized.astype('float32')

        imgLab = cv2.cvtColor(imgNormalized, cv2.COLOR_RGB2Lab)
        channelL = imgLab[:,:,0]
        
        imgLabResized = cv2.cvtColor(cv2.resize(imgNormalized,(224,224)), cv2.COLOR_RGB2Lab)
        channelLResized = imgLabResized[:,:,0]
        channelLResized -= 50

        self.colorModel.setInput(cv2.dnn.blobFromImage(channelLResized))
        result = self.colorModel.forward()[0,:,:,:].transpose((1,2,0))

        resultResized = cv2.resize(result, (self.width,self.height))
        
        self.ImgOUT = np.concatenate((channelL[:,:,np.newaxis], resultResized), axis=2)
        self.ImgOUT = np.clip(cv2.cvtColor(self.ImgOUT, cv2.COLOR_LAB2BGR),0,1)
        self.ImgOUT = np.array((self.ImgOUT)*255, dtype=np.uint8)

        self.ImgFinal = np.hstack((self.img, self.ImgOUT))
    
   

    