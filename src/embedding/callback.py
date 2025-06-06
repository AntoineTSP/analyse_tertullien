from tensorflow.keras.callbacks import Callback


# Define a callback to capture logs
class TrainingLogger(Callback):
    def __init__(self):
        super().__init__()
        self.logs = []

    def on_epoch_end(self, epoch, logs=None):
        # logs is a dict with metrics like loss, accuracy, val_loss, etc.
        self.logs.append({"epoch": epoch + 1, **logs})
