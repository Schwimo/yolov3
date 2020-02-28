#! /usr/bin/env Python3

import argparse
import os
from sys import platform

import cv2

from yolov3.models import * # set ONNX_EXPORT in models.py
from yolov3.utils.datasets import *
from yolov3.utils.utils import *

this_file_dir = os.path.dirname(os.path.realpath(__file__)) + '/../../'

class Detector:

    def __init__(self, callback = None,
                       cfg = this_file_dir + 'cfg/yolov3.cfg',
                       names = this_file_dir + 'data/obj.names',
                       weights = this_file_dir + 'weights/sofarthebest.pt',
                       source = None, 
                       output = this_file_dir + 'output',
                       img_size = 416,
                       conf_thres = 0.1,
                       iou_thres = 0.2,                       
                       fourcc = 'mp4v',
                       half = True,                       
                       device = "",
                       view_img = True,
                       save_txt = False,
                       classes = 0,
                       agnostic_nms = True,
                       classify = False):
        """
        Initialize the YoloV3 class
        """

        # The callback function to pass the detected image back to the customer logic
        self.callback = callback             

        self.cfg = cfg
        self.names = names
        self.weights = weights        
        self.source = this_file_dir + 'data/samples' if source is None else source        
        self.output = output
        self.img_size =  (320, 192) if ONNX_EXPORT else img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.fourcc = fourcc
        self.half = half
        self.device = device
        self.view_img = view_img
        self.save_txt = save_txt
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.classify = classify

        self.vid_path = None
        self.vid_writer = None
        
    def convert_to_onnx(self):
        """
        Convert the model to onnx
        """

        pass

    def detect(self, save_img=False):

        with torch.no_grad():
          
            webcam = self.source == '0' or self.source.startswith('rtsp') or self.source.startswith('http') or self.source.endswith('.txt')
            custom_cam = self.source.startswith('realsense2')

            # Initialize
            device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else self.device)
            if os.path.exists(self.output):
                shutil.rmtree(self.output)  # delete output folder
            os.makedirs(self.output)  # make new output folder

            # Initialize model
            model = Darknet(self.cfg, self.img_size)

            # Load weights
            attempt_download(self.weights)
            if self.weights.endswith('.pt'):  # pytorch format
                model.load_state_dict(torch.load(self.weights, map_location=device)['model'])
            else:  # darknet format
                load_darknet_weights(model, self.weights)

            # Second-stage classifier            
            if self.classify:
                modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
                modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
                modelc.to(device).eval()
            
            # Eval mode
            model.to(device).eval()

            # Export mode
            if ONNX_EXPORT:
                model.fuse()
                img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
                f = self.opt.weights.replace(self.opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
                torch.onnx.export(model, img, f, verbose=False, opset_version=11)

                # Validate exported model
                import onnx
                model = onnx.load(f)  # Load the ONNX model
                onnx.checker.check_model(model)  # Check that the IR is well formed
                print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
                return

            # Half precision
            self.half = self.half and device.type != 'cpu'  # half precision only supported on CUDA
            if self.half:
                model.half()

            # Set Dataloader            
            if webcam:                
                dataset = LoadStreams(self.source, img_size=self.img_size)                
            elif custom_cam:                
                torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = self.source
            else:                
                dataset = LoadImages(self.source, img_size=self.img_size)

            # Get names and colors
            names = load_classes(self.names)            
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

            # Run inference
            t0 = time.time()
            for path, img, im0s, vid_cap in dataset:
                t = time.time()

                 # Get detections
                img = torch.from_numpy(img).to(device)
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                pred = model(img)[0]

                if self.half:
                    pred = pred.float()

                # Apply NMS                
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)

                # Apply Classifier
                if self.classify:
                    pred = apply_classifier(pred, modelc, img, im0s)

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if webcam or custom_cam:  # batch_size >= 1
                        p, s, im0 = path[i], '%g: ' % i, im0s[i]                    
                    else:
                        p, s, im0 = path, '', im0s

                    save_path = str(Path(self.output) / Path(p).name)
                    s += '%gx%g ' % img.shape[2:]  # print string
                    
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += '%g %ss, ' % (n, names[int(c)])  # add to string

                        # Write results
                        for *xyxy, conf, cls in det:
                            if self.save_txt:  # Write to file
                                with open(save_path + '.txt', 'a') as file:
                                    file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                            if save_img or self.view_img:  # Add bbox to image
                                label = '%s %.2f' % (names[int(cls)], conf)
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                            
                        self.callback(det)

                    # Print time (inference + NMS)
                    #print('%sDone. (%.3fs)' % (s, time.time() - t))

                    # if self.callback:
                    #     self.callback()

                    # Stream results                                    
                    if self.view_img:
                        cv2.imshow(p, im0)
                        if cv2.waitKey(1) == ord('q'):  # q to quit
                            raise StopIteration

                    # Save results (image with detections)
                    if save_img:
                        if dataset.mode == 'images':
                            cv2.imwrite(save_path, im0)
                        else:
                            if vid_path != save_path:  # new video
                                vid_path = save_path
                                if isinstance(vid_writer, cv2.VideoWriter):
                                    vid_writer.release()  # release previous video writer

                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*self.opt.fourcc), fps, (w, h))
                            vid_writer.write(im0)

            if self.save_txt or save_img:
                print('Results saved to %s' % os.getcwd() + os.sep + self.output)
                if platform == 'darwin':  # MacOS
                    os.system('open ' + self.output + ' ' + save_path)

            #print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    self.opt = parser.parse_args()    

    with torch.no_grad():
        detect()
