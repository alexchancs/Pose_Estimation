
import sys
from pose_video_capture import *
from live_view import *

if __name__ == '__main__':
    vparser = PoseVideoCapture.argumentParser()
    parser = argparse.ArgumentParser(parents=[vparser], description='Pose Demo')
    args = parser.parse_args()

    print()
    print()
    print('------------ Starting Pose Analyzer ------------')
    print()

    video_pose_capture = PoseVideoCapture(args)
    
    view = LiveView(video_pose_capture.live_view_queue)
    video_pose_capture.start()
    view.start()
    loop = True
    
    while loop:
        repeat = True
        if repeat:
            repeat = True
            if video_pose_capture.ispaused():
                print("video_pose_capture")
            key = input()
            if 'x' in key:
                loop = False
                repeat = False
                break
    print("Stopping")
    view.stop()
    video_pose_capture.stop()
    view.join()
    sys.exit(0)
