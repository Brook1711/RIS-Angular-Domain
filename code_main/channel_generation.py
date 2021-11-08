import numpy as np

class User():
    def __init__(self, postion = [0, 0, 0], ant_type = 'ULA', ant_num = 32, name = 0) -> None:
        self.position = np.array(postion)
        self.ant_type = ant_type
        self.ant_num = ant_num
        self.name = name

        # omega information
        self.phi_channel_support = np.array([0 for a in self.ant_num])
    
class BS(): 
    def __init__(self, postion = [10, 0, 20], ant_type = 'ULA', ant_num = 64) -> None:
        self.position = np.array(postion)
        self.ant_type = ant_type
        self.ant_num = ant_num

class RIS(): 
    def __init__(self, postion = [0, 10, 10], ant_type = 'ULA', ant_num = 64, K = 4) -> None:
        # basic information
        self.position = np.array(postion)
        self.ant_type = ant_type
        self.ant_num = ant_num

        # AoA/AoD information
        self.varphi_channel_support_common = np.array([0 for i in range(self.ant_num)])
        self.varphi_channel_support_all_users = np.array([[0 for m in range(self.ant_num)] for k in range(K)])
        self.omega_channel_support = np.array([0 for m in range(self.ant_num)])

class Evn():
    def __init__(self, K = 4) -> None:
        np.random.seed(1234)
        self.BS = BS()
        self.RIS = RIS(K=K)
        self.user_list = [User(name = i) for i in range(K)]
        