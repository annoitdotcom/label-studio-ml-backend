from dcnet.trainer.utilities.signal_monitor import SignalMonitor


class FileMonitorLearningRate(object):
    """

    """

    def __init__(self, opt):
        self.monitor = SignalMonitor(opt.logging.signal_file_name)

    def get_learning_rate(self, epoch, step):
        signal = self.monitor.get_signal()

        if signal is not None:
            return float(signal)
        return None
