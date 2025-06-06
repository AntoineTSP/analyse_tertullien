from io import StringIO

from tensorflow.keras import Model


def get_model_summary(model: Model) -> str:
    stream = StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    summary_str = stream.getvalue()
    stream.close()
    return summary_str
