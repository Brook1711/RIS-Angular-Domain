import numpy as np

class User():
    def __init__(self, postion = [0, 0, 0], ant_type = 'ULA', ant_num = 32) -> None:
        self.position = np.array(postion)
        self.ant_type = ant_type
        self.ant_num = ant_num
    
class BS(): 
    def __init__(self, postion = [10, 0, 20], ant_type = 'ULA', ant_num = 64) -> None:
        self.position = np.array(postion)
        self.ant_type = ant_type
        self.ant_num = ant_num

class RIS(): 
    def __init__(self, postion = [0, 10, 10], ant_type = 'ULA', ant_num = 64) -> None:
        self.position = np.array(postion)
        self.ant_type = ant_type
        self.ant_num = ant_num