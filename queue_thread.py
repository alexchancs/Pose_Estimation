from threading import Thread
from queue import Queue
import logging


class NoneJobError(Exception):
    pass


class QueueThread(Thread):
    def __init__(self, queue: Queue, name: str = 'Untitled'):
        Thread.__init__(self)
        self.loop = False
        self.name = name
        self.queue = queue
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)

    def job(self):
        raise NoneJobError('Job function is not implemented')

    def run(self) -> None:
        self.logger.info('Thread started')
        self.loop = True
        while self.loop:
            self.job()
            self.queue.task_done()

        self.logger.info('Thread stopped')

    def stop(self):
        self.loop = False

