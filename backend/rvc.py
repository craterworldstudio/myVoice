from core.stage import Stage
from backend.loader import RVCLoader

class RVC(Stage):
    def __init__(self,
                 path_pth
                 
                 ):
        super().__init__("rvc")
        self.path_pth = path_pth
        self.loader = RVCLoader(path_pth)
        self.model = self.loader.load()
        

    def process(self, audio):

        


        return audio