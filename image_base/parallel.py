from multiprocessing import Queue, Process, cpu_count
import abc

class TargetWrapper:

    def __init__(self, func, readfrom, writeto=None, returns=True):
        self.readfrom = readfrom
        self.writeto = writeto
        self.func = func
        self.processing = False
        self.returns = returns


    def __call__(self):

        self.processing = True
        while self.processing:
            items = self.readfrom.get()
            if items[0]:
                if self.returns:
                    result = self.func(**items[1])

                    self.writeto.put(result)
                else:
                    self.func(**items[1])
            else:
                self.processing = False

        self.readfrom.close()
        self.writeto.close()


class Feeder:
    """ Worker which does work in another process to make it ready for the current process when requested
        Calls to the iterator apply block until a new result is available.
    """

    def __init__(self, worker, n_jobs=-1):
        self.worker = worker
        if n_jobs == -1:
            n_jobs = cpu_count()
        self.n_jobs = n_jobs

    def apply(self, iterator):
        self.readfrom = Queue()
        self.writeto = Queue()
        processes = [Process(target=TargetWrapper(self.worker, self.writeto, self.readfrom), name='FeederPool {0}'.format(i)) for i in range(self.n_jobs)]
        for process in processes:
            process.start()

        outbound = 0
        for item in iterator:
            if outbound == self.n_jobs:
                result = self.readfrom.get(True)
                outbound -= 1
                yield result

            self.writeto.put((True, item))
            outbound += 1

        while outbound > 0:
            result = self.readfrom.get(True)
            yield result
            outbound -=1

        for _ in processes:
            self.writeto.put((False,))
        self.readfrom.close()
        self.readfrom.join_thread()
        self.writeto.close()
        self.writeto.join_thread()

class Eater:
    """ Worker which does work in another process so that the current process can continue.  Calls to apply can not
    return anything and return immediately.
    """

    def __init__(self, worker, n_jobs=-1, max_size=-1):
        self.worker = worker
        if n_jobs == -1:
            n_jobs = cpu_count()
        self.n_jobs = n_jobs
        self.max_size = max_size

    def apply(self, iterator):
        self.writeto = Queue()
        processes = [Process(target=TargetWrapper(self.worker, self.writeto, returns=False), name='EaterPool {0}'.format(i)) for i in range(self.n_jobs)]
        for process in processes:
            process.start()

        for item in iterator:
            self.writeto.put((True, item))

        for _ in processes:
            self.writeto.put((False,))
        self.writeto.close()
        self.writeto.join_thread()
