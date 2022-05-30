import os
import requests
import numpy as np
import cv2
import torch
import torchvision
from argparse import ArgumentParser
from pytorch_lightning import LightningModule
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_320_fpn
from time import time
from config import pretrained_model_dir


class FasterRCNNModelEval(LightningModule):
    def __init__(self, num_classes=4):
        super().__init__()
        self.save_hyperparameters()
        #if is_mobile:
        #    self.model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
        #else:
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
        #self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, x):
        self.model.eval()
        return self.model(x)
    
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

#for param in model.parameters():
#    param.grad = None

class_labels = {0:'background', 1:'Summit', 2:'Coke', 3:'Pine Juice'}
class_colors = {0:(0,0,0), 1:(255,102,102), 2:(51,51,255), 3:(51,255,51)}


def run_video_demo(model, camera, record, filename):
    print('Select window and press q to quit.')
    cap = cv2.VideoCapture(camera)
    #w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #print(f"w: {w}, h: {h}")
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 128)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 96)
    videowriter = None
    if record:
        videowriter = cv2.VideoWriter(filename,
                                    cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                    10,
                                    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), 
                                    isColor=True)
    with torch.no_grad():
        prev_frame_time = 0
        new_frame_time = 0
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Error: failed to capture frame")
                break

            #image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            image = torchvision.transforms.ToTensor()(image).cuda()
            predictions = model([image])[0]
            
            for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
                label = label.item()
                if label == 0 or score < 0.95:
                    continue
                minX, minY, maxX, maxY = box.int().tolist()
                #print(minX, minY, maxX, maxY, label, score.item())
                cv2.rectangle(frame, (minX, minY), (maxX, maxY), class_colors[label], 2)
                cv2.putText(frame, f"{class_labels[label]} {score.item():.2f}", (minX, minY-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors[label], 2)
            
            if args.show_fps:
                new_frame_time = time()
                fps = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time
                cv2.putText(frame, str(int(fps)), (4, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow('frame',frame)
            if videowriter is not None:
                if videowriter.isOpened():
                    videowriter.write(frame)
            if cv2.waitKey(1) == ord('q'):
                break

            #print(f'fps: {1/(time()-loop_time)}')

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = ArgumentParser(description="video demo for Drinks dataset object detection")
    parser.add_argument("--camera",
                        default=0,
                        type=int,
                        help="Camera index")
    parser.add_argument("--record",
                        default=False,
                        action='store_true', 
                        help="Record video")
    parser.add_argument("--filename",
                        default="demo.mp4",
                        help="Video filename")
    parser.add_argument("--model-path",
                        default=pretrained_model_dir,
                        help="Path to model")
    parser.add_argument("--show-fps",
                        default=False,
                        action='store_true', 
                        help="Display fps")

    args = parser.parse_args()

    if not os.path.exists(pretrained_model_dir):
        os.makedirs(os.path.dirname(pretrained_model_dir), exist_ok=True)
        fname = os.path.basename(pretrained_model_dir)
        url = f'https://github.com/DJSapit/Drinks-Dataset-Faster-RCNN-Sapit/releases/download/v0.1.0-alpha/{fname}'
        print(f'downloading pretrained model from {url}')
        r = requests.get(url, allow_redirects=True)
        with open(pretrained_model_dir, 'wb') as file:
            file.write(r.content)
        if os.path.exists(pretrained_model_dir):
            print("pretrained model downloaded successfully")
        else:
            raise Exception("pretrained model download failed")

    model = FasterRCNNModelEval(num_classes=4)
    model = model.load_from_checkpoint(args.model_path)
    if torch.cuda.is_available():
        model.cuda().eval()
        run_video_demo(model=model,
                    camera=args.camera,
                    record=args.record,
                    filename=args.filename)
    else:
        print("Error: no CUDA device available")