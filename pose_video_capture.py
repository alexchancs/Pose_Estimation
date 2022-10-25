from multiprocessing import Event
from threading import Thread, Event
from queue import Queue
import cv2
import argparse
import time
from pose_capture import *
import logging


class PoseVideoCaptureError(Exception):
    pass


class PoseVideoCaptureEOSError(PoseVideoCaptureError):
    pass


class PoseVideoCaptureDeviceError(PoseVideoCaptureError):
    pass


class PoseVideoCapture(Thread):
    def __init__(self, args):
        Thread.__init__(self)

        self.model = PoseCaptureModel(args.model, args.task)

        # Open the camera device
        self.capture = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
        if self.capture.isOpened() is False:
            raise PoseVideoCaptureDeviceError('Camera %d could not be opened.' % (args.camera))

        # Set the capture parameters
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

        if args.mjpg:
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        if args.fps is not None:
            self.capture.set(cv2.CAP_PROP_FPS, args.fps)

        # Default to RGB
        self.capture.set(cv2.CAP_PROP_CONVERT_RGB, 1)

        # Get the actual frame size
        # Not work for OpenCV 4.1
        self.width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        self.actual_fps = 0.0
        self.target_fps = args.target_fps

        self.start_paused = False
        self.capture_loop = False
        self.live_view = True
        self.live_view_queue = Queue()
        self.process_queue = Queue()
        
        self.logger = logging.getLogger('PoseVideoCapture')
        self.logger.setLevel(logging.INFO)
        self.event = Event()
        
    def run(self) -> None:
        self.logger.info('Thread Started')
        self.capture_loop = True
        if self.start_paused:
            self.event.clear()
        else:
            self.event.set()
        while self.capture_loop:
            self.event.wait()
            
            t1 = time.time()
            _, frame = self.capture.read()
            if frame is None:
                break
            humans = self.model.process(frame)
            self.live_view_queue.put_nowait((frame, humans))
            self.process_queue.put_nowait(humans)
            t2 = time.time() - t1
            time_delay = (1 / self.target_fps) - t2
            if time_delay > 0:
                time.sleep(time_delay)
        self.capture.release()
        self.logger.info('Thread Stopped')
    
    def toggle_button(self):
        if self.event.is_set():
            self.pause()
        else:
            self.resume()
    
    def pause(self):
        self.event.clear()
        print('Pause Video')
    
    def resume(self):
        self.process_queue.queue.clear()
        self.live_view_queue.queue.clear()
        
        self.event.set()
        print('Resume Video')
    
    def stop(self):
        self.capture_loop = False
        self.event.set()
        self.join()
        
    def ispaused(self):
        return not self.event.is_set()

    @staticmethod
    def argumentParser(**kwargs):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--camera', '-c', type=int, default=0, metavar='CAMERA_NUM', help='Camera number')
        parser.add_argument('--width', type=int, default=1280, metavar='WIDTH', help='Capture width')
        parser.add_argument('--height', type=int, default=720, metavar='HEIGHT', help='Capture height')
        parser.add_argument('--fps', type=int, default=-1, metavar='FPS', help='Capture frame rate')
        parser.add_argument('--target-fps', type=int, default=60, metavar='TARGET_FPS', help='Target frame rate')
        parser.add_argument('--mjpg', action='store_true', help='If set, capture video in motion jpeg format')
        parser.add_argument('--nodrop', action='store_true', help='If set, disable frame drop feature')
        parser.add_argument('--golden_data', type=str, default='skeleton_data/all_skeleton_data_20220919_152744.json', metavar='SKELETON_DATA', help='skeleton_data')

        parser.set_defaults(**kwargs)
        parser = PoseCaptureModel.add_parse_argument(parser)
        return parser
