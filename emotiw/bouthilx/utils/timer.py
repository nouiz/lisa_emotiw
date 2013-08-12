import time
import datetime
import sys

def seconds_to_string(seconds):
    return str(datetime.timedelta(seconds=seconds)).split('.')[0]

class Timer(object):
    def __init__(self,op_todo=None,update_freq=5,out=None,min_time=120):
        self.op_todo = op_todo
        self.start_time = None
        self.update_time = None
        self.op_done = 0
        self.out = out
        self.min_time = min_time

    def start(self):
        self.start_time = time.clock()
        self.update_time = time.clock()

    def reset_op_todo(self,op_todo):
        self.op_todo = op_todo

    def update(self,op_done):
        if self.start_time==None:
            raise BaseException("Need to start the timer before calling updates")

        self.op_done += op_done

        if time.clock() - self.update_time > self.min_time:
            self.update_time = time.clock()

            return (time.clock()-self.start_time)/(0.+self.op_done)*(self.op_todo-self.op_done)
        
        return None

    def print_update(self,op_done):
        update = self.update(op_done)
        if update and self.out:
            self.out.write("\rtime remaining : "+seconds_to_string(update))
            self.out.flush()
        elif update:
            sys.stdout.write("\rtime remaining : "+seconds_to_string(update))
            sys.stdout.flush()

    def over(self):
        if self.start_time==None:
            raise BaseException("Need to start the timer before ending it")

        return time.clock()-self.start_time
