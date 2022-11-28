
class ModelWrapper:

    def __init__(self, model):
        self.model = model


    def translate_and_predict(self, instance):
        # Do translation
        self.model.predict()
