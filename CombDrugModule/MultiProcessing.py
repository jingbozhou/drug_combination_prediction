import time, signal
from multiprocessing import cpu_count, Process, Queue, Pool

class MultiProcessing(object):
    """Class to run jobs in an d manner.
    You would use this class to run several jobs on a local computer that has
    several cpus.
    ::
        t = MultiProcessing(maxcpu=2)
        t.add_job(func, func_args)
        t.run()
        t.results[0] # contain returned object from the function *func*.

    .. warning:: the function must be a function, not a method. This is inherent
        to multiprocess in the multiprocessing module.

    .. warning:: the order in the results list may not be the same as the
        list of jobs. see :meth:`run` for details
    """
    def __init__(self, maxcpu=None):
        """
        :param maxcpu: default returned by multiprocessing.cpu_count()
        """
        if maxcpu is None:
            maxcpu = cpu_count()
            
        self.maxcpu = maxcpu
        self.reset()
        
    def reset(self):
        """remove joves and results"""
        self.jobs = []
        self.results = Queue()
        
    def add_job(self, func, *args, **kargs):
        """add a job in the pool"""
        t = Process(target=func, args=args, kwargs=kargs)
        self.jobs.append(t)
        
    def _cb(self, results):
        self.results.append(results)
        
    def run(self, delay=1e-8):
        def init_worker():
            signal.signal(signal.SIGINT, signal.SIG_IGN)

        self.results = []
        self.pool = Pool(self.maxcpu, init_worker)
        
        for process in self.jobs:
            self.pool.apply_async(process._target, process._args,
                                  process._kwargs, callback=self._cb)

            # ensure the results have same order as jobs
            # maybe important if you expect the order of the results to 
            # be the same as inut; otherwise set delay to 0 
            time.sleep(delay)
        
        try:
            while True:
                time.sleep(1)
                # check if all processes are finished.
                # if so, finished.
                count = len(self.results)
                if count == len(self.jobs):
                    break

        except KeyboardInterrupt:#pragma: no cover
            print("\nCaught interruption. " + 
            "Terminating the Pool of processes... ",)
            self.pool.terminate()
            self.pool.join()
            print("... done")
        else:
            # Closing properly the pool
            self.pool.close()
            self.pool.join()

        # Pool cannot be pickled. So, if we want to pickel "MultiProcessing"
        # class itself, we must desctroy this instance
        del self.pool 

        self.finished = True
