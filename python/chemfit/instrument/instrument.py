class Instrument():
    def __init__(self):
        self.settings = None

    def get_default_settings(self, original_settings = {}):
        raise NotImplementedError()