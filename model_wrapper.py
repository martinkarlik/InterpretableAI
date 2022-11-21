
import model


class ModelWrapper:

    def __init__(self):
        self.model = model.CNN_model()


    def predict_and_translate(self):
        # Do trans
        self.model.predict()