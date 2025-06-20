from tensorflow.keras.callbacks import Callback


# Define a callback to capture logs
class TrainingLogger(Callback):
    def __init__(self):
        super().__init__()
        self.logs = []

    def on_epoch_end(self, epoch, logs=None) -> None:
        """
        Called at the end of an epoch. Captures logs and appends them to self.logs.

        Args:
            epoch (int): The index of the epoch.
            logs (dict, optional): A dictionary containing metrics like loss, accuracy, etc.
        """

        # logs is a dict with metrics like loss, accuracy, val_loss, etc.
        self.logs.append({"epoch": epoch + 1, **logs})
