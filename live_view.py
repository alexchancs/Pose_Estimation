from queue import Queue
import cv2
import datetime
import numpy as np
import time
from queue_thread import QueueThread


class LiveView(QueueThread):
    def __init__(self, queue: Queue):
        QueueThread.__init__(self, queue, 'LiveView')
        self.fpsCounter = IntervalCounter(10)
        self.windowName = 'LiveView'

    def post_job(self):
        cv2.destroyAllWindows()

    def job(self) -> None:
        frame, humans = self.queue.get(block=True)
        # frame = cv2.UMat(frame)
        for human in humans:
            for part in human.body_part:
                position = human.position(part)
                if position >= (0, 0):
                    color = (255, 0, 0)
                    if 'left' in part:
                        color = (0, 255, 0)
                    elif 'right' in part:
                        color = (0, 0, 255)
                    cv2.circle(frame, position, 3, color, 2)

                    if 'center_hip' in part:
                        pt1 = (position[0], 0)
                        pt2 = (position[0], frame.shape[0])
                        cv2.line(frame, pt1, pt2, (255, 255, 255), 2)

            for line in human.lines:
                color = (255, 255, 255)
                pt1, pt2 = line
                cv2.line(frame, pt1, pt2, color, 2)

        interval = self.fpsCounter.measure()
        if interval is not None:
            fps = 1.0 / interval
            dt = datetime.datetime.now().strftime('%F %T')
            fpsInfo = '{0}{1:.2f} {2}'.format('FPS:', fps, dt)
            cv2.putText(frame, fpsInfo, (8, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow(self.windowName, frame)
        cv2.waitKey(1)


class IntervalCounter:
    """A counter to measure the interval between the measure method calls.

    Attributes:
        numSamples: Number of samples to calculate the average.
        samples: Buffer to store the last N intervals.
        lastTime: Last time stamp
        count: Total counts
    """

    def __init__(self, numSamples):
        """
        Args:
            numSamples(int): Number of samples to calculate the average.
        """
        self.numSamples = numSamples
        self.samples = np.zeros(self.numSamples)
        self.lastTime = time.time()
        self.count = 0

    def __del__(self):
        pass

    def measure(self):
        """Measure the interval from the last call.

        Returns:
            The interval time count in second.
            If the number timestamps captured in less than numSamples,
            None will be returned.
        """
        curTime = time.time()
        elapsedTime = curTime - self.lastTime
        self.lastTime = curTime
        self.samples = np.append(self.samples, elapsedTime)
        self.samples = np.delete(self.samples, 0)
        self.count += 1
        if self.count > self.numSamples:
            return np.average(self.samples)
        else:
            return None
