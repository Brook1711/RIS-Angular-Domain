import numpy as np
import cmath
import matplotlib.pyplot as plt

def D_N(N, x):
    if abs(x) == 0:
        return N
    else:
        return 1/np.sqrt(N)*cmath.exp(1j*x*(N-1)/2)*(cmath.sin((N)*x/2))/(cmath.sin(x/2))

class User():
    def __init__(self, postion = [0, 0, 0], ant_type = 'ULA', ant_num = 32, name = 0) -> None:
        self.position = np.array(postion)
        self.ant_type = ant_type
        self.ant_num = ant_num
        self.name = name

        # omega information
        self.phi_channel_support = np.array([0 for a in range(self.ant_num)])
    
class BS(): 
    def __init__(self, postion = [10, 0, 20], ant_type = 'ULA', ant_num = 64) -> None:
        self.position = np.array(postion)
        self.ant_type = ant_type
        self.ant_num = ant_num

        # psi information
        self.psi_channel_support = np.array([0 for a in range(self.ant_num)])

class RIS(): 
    def __init__(self, postion = [0, 10, 10], ant_type = 'ULA', ant_num = 64, K = 4) -> None:
        # basic information
        self.position = np.array(postion)
        self.ant_type = ant_type
        self.ant_num = ant_num
        self.K = K

        # AoA/AoD information
        self.varphi_channel_support_common = np.array([0 for i in range(self.ant_num)])
        self.varphi_channel_support_all_users = np.array([[0 for m in range(self.ant_num)] for k in range(K)])
        self.omega_channel_support = np.array([0 for m in range(self.ant_num)])

        ## delta information
        delta_lower_bound = -0.5 * 1 / self.ant_num
        delta_upper_bound = 0.5  * 1 / self.ant_num
        self.delta_omega = np.random.uniform(delta_lower_bound, delta_upper_bound, self.ant_num)
        self.delta_varphi = np.random.uniform(delta_lower_bound, delta_upper_bound, self.ant_num)
        self.RIS_grid = np.array([(m-1)/self.ant_num - 0.5 for m in range(self.ant_num)])

        self.omega = np.array([-1.0 for m in range(self.ant_num)])
        self.varphi = np.array([[-1.0 for m in range(self.ant_num)] for k in range(self.K)])

        # self.generate_angle()

    def generate_angle(self):
        for i, s in enumerate(self.omega_channel_support):
            if s == 1:
                self.omega[i] = (self.RIS_grid[i] - self.delta_omega[i])
            

        for k in range(self.K):
            for m, s in enumerate(self.varphi_channel_support_all_users[k]):
                if s == 1:
                    self.varphi[k][m] = (self.RIS_grid[m] - self.delta_varphi[m])

        return self.omega, self.varphi

class Evn():
    def __init__(self, K = 4) -> None:
        np.random.seed(1234)
        self.BS = BS()
        self.RIS = RIS(K=K)
        self.user_list = [User(name = i) for i in range(K)]

        # initialize the channel
        self.channel_init()
    
    def channel_init(self):
        p10 = 0.3
        p01 = 0.1
        p0 = p10/(p10+p01)
        p1 = 1-p0

        # generate Markov chain
        tmp_support_omega = np.array([0.0 for m in range(self.RIS.ant_num)])
        if np.random.uniform(0,1,1)[0] < p0:
            tmp_support_omega[0] = 0
        for i in range(self.RIS.ant_num - 1):
            if tmp_support_omega[i] == 0:
                if np.random.uniform(0,1,2)[0] < p01:
                    tmp_support_omega[i+1] = 1
                else:
                    tmp_support_omega[i+1] = 0
            else:
                if np.random.uniform(0,1,2)[1] < p10:
                    tmp_support_omega[i+1] = 0
                else:
                    tmp_support_omega[i+1] = 1
        plt.scatter(range(self.RIS.ant_num), tmp_support_omega)
        plt.plot(range(self.RIS.ant_num), tmp_support_omega)
        plt.show()
        print(0)
#%%
E0 = Evn()
# %%
