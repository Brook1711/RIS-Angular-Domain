import numpy as np
import cmath
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.special import gamma as gamma_function
from scipy.special import digamma

alpha_0, alpha_1 = 1, 1
beta_0, beta_1 = 0.001, 1
K=4
p01 = 0.5
p10 = 0.5
trans_pro_C = {
    'p01':p01,
    'p10':p10,
    'p1':p01/(p01+p10)
    }
M = 64
tau = 8

class E_step():
    def __init__(self, FK, kappa, YK, trans_pro_C, trans_prob_CS):
        # the parameters from M, including: 
        ## used in module A
        self.FK = FK
        self.kappa = kappa
        self.YK = YK

        ## used in module B
        self.trans_pro_C = trans_pro_C          # p(c_m+1 | c_m)
        self.trans_pro_CS = trans_prob_CS       # p(s_k,m | c_m)
        
        self.input_from_M = 0

        self.FB_MP_loops = 10
        self.VBI_loops = 10

        # preperation
        ## parameters of q(x)
        self.Sigma = [np.mat(np.zeros((M,M), dtype = np.complex128)) for k in range(K)]          # is a list of complex matrix, K X (M, M)
        self.Mu = [np.zeros((M), dtype = np.complex128) for k in range(K)]             # is a list of complex vecter, K X (M, )
        ## parameters of q(gamma)
        self.widetilde_ab = [np.mat(np.zeros((M,2) ) ) for k in range(K)]   # is a list of real-valued matrix, K X (M, 2)
        ## parameters of q(s), postierior
        self.widetilde_pi = [np.zeros((M)) for k in range(K)]   # is a list of real-valued vector, K X (M, )
        ## parameters of p(s), prior               
        self.prior_pi = [np.array([0.5 for m in range(M)]) for k in range(K)]       # is a list of real-valued vector, K X (M, )
        ## the massages passing component
        self.bar_nu_h_to_s = [np.zeros((M)) for k in range(K)]      # is a list of real-valued vector, K X (M, ), B pass to A
        self.bar_nu_eta_to_s = [np.zeros((M)) for k in range(K)]    # is a list of real-valued vector, K X (M, ), A pass to B
        self.Forward_messages = np.array([0 for m in range(M)])
        self.Backward_messages = np.array([0 for m in range(M)])
        
        pass

    def begin(self):
        # An outer-iterration 
        for FB_MP_step in range(self.FB_MP_loops):
            # An inter-iterration
            self.initialization_of_module_A(FB_MP_step)
            for VBI_step in range(self.VBI_loops):
                self.VBI_module_A()
            self.bar_nu_h_to_s = self.MP_module_B()

    def initialization_of_module_A(self, outer_iteration_index):
        # in the first outer iteration, initialize the postierior to be the prior
        if outer_iteration_index == 0:
            # initialize q(s)
            for k in range(K):
                self.widetilde_pi[k] = self.prior_pi[k]
            # initialize q(gamma)
            for k in range(K):
                for m in range(M):
                    self.widetilde_ab[k][m][0] = self.widetilde_pi[k][m] * alpha_1 + (1-self.widetilde_pi[k][m]) * alpha_0
                    self.widetilde_ab[k][m][1] = self.widetilde_pi[k][m] * beta_1 + (1-self.widetilde_pi[k][m]) * beta_0
            # initialize q(x)
            for k in range(K):
                diag_widetilde_a_div_b = np.diag(np.array([self.widetilde_ab[k][m][0]/self.widetilde_ab[k][m][1] for m in range(M)]))
                self.Sigma = np.linalg.inv(diag_widetilde_a_div_b + self.FK[k].H *np.diag(self.kappa)*self.FK[k])
                self.Mu = self.Sigma * self.FK[k].H * np.diag(self.kappa) * self.YK[k]
        else:
            # do nothing, since the q functions have already been updated in the previous outer iteration
            pass
        pass

    def VBI_module_A(self, bar_nu_h_to_s):
        # the original (last inner-iteration)
        last_inner_Sigma = self.Sigma.copy()
        last_inner_Mu = self.Mu.copy()

        last_inner_widetilde_ab = self.widetilde_ab.copy()
        # first update q(x)
        for k in range(K):
            diag_widetilde_a_div_b = np.diag(np.array([self.widetilde_ab[k][m][0]/self.widetilde_ab[k][m][1] for m in range(M)]))
            self.Sigma = np.linalg.inv(diag_widetilde_a_div_b + self.FK[k].H *np.diag(self.kappa)*self.FK[k])
            self.Mu = self.Sigma * self.FK[k].H * np.diag(self.kappa) * self.YK[k]
        # second update q(gamma)
        for k in range(K):
            for m in range(M):
                self.widetilde_ab[k][m][0] = self.widetilde_pi[k][m] * alpha_1 + (1 - self.widetilde_pi[k][m]) * alpha_0 + 1
                self.widetilde_ab[k][m][1] = np.power(np.abs(last_inner_Mu[k][m]),2) + last_inner_Sigma[k][m,m] + self.widetilde_pi[k][m] * beta_1 + (1- self.widetilde_pi[k][m]) * beta_0
        # third update q(s)
        for k in range(K):
            for m in range(M):
                C1 = self.prior_pi[k][m] * np.power(beta_1, alpha_1)/gamma_function(alpha_1) * np.exp((alpha_1 - 1)*(digamma(last_inner_widetilde_ab[k][m][0]) - np.log(last_inner_widetilde_ab[k][m][1]) )-beta_1 * last_inner_widetilde_ab[k][m][0]/last_inner_widetilde_ab[k][m][1] )
                C0 = (1 - self.prior_pi[k][m]) * np.power(beta_0, alpha_0)/gamma_function(alpha_0) * np.exp((alpha_0 - 1)* (digamma(last_inner_widetilde_ab[k][m][0]) - np.log(last_inner_widetilde_ab[k][m][1]) ) - beta_0 * last_inner_widetilde_ab[k][m][0]/last_inner_widetilde_ab[k][m][1] )
                self.widetilde_pi = C1/(C1+C0)

        # update messages passed to B
        for k in range(K):
            for m in range(M):
                nu_1 = self.widetilde_pi[k][m]/self.bar_nu_h_to_s[k][m]
                nu_0 = (1-self.widetilde_pi[k][m])/(1-self.bar_nu_h_to_s[k][m])
                self.bar_nu_eta_to_s[k][m] = nu_1 / (nu_1 + nu_0)
        # bar_nu_eta_to_s is used in the F/B MP, i.e., module B for prior information caculation.
        return 0

    def MP_module_B(self, bar_nu_eta_to_s):
        # input: bar_nu_eta_to_s, is a numpy array, shape(K, M), value in {0, 1}
        # nu_in = self.bar_nu_eta_to_s.copy()
        # output: bar_nu_h_to_s, is a numpy array, shape(K, M), value in {0, 1}
        ## output initialization
        # bar_nu_h_to_s = np.array(np.zeros(K, M))
        self.update_alpha()
        self.update_beta()
        # calculate the first m nu_out
        condition_list_m1_except_k = []
        for i in range(2**(K-1+1)):
            condition_str = ('{:0'+str(K-1+1)+'b}').format(i)
            condition_list = np.array([int(j) for j in condition_str])
            condition_list_m1_except_k.append(condition_list)
        for k in range(K):
            prepera_to_sum_res_1 = []
            prepera_to_sum_res_0 = []
            for condition in condition_list_m1_except_k:
                condition_res_1 = np.insert(condition, k+1, 1)
                func_gm1_res_1 = self.g_m(m=0, C_m= condition_res_1[0], C_m1=0, S_km=condition_res_1[1,:])
                beta_Cm = condition_res_1[0] * self.Backward_messages[0] + (1- condition_res_1[0]) * (1- self.Backward_messages[0])
                nu_in_K1 = np.array([self.bar_nu_eta_to_s[k][0] for k in range(K)])
                cumprod_lambda_except_k = np.cumprod(np.delete(condition_res_1[1,:] * nu_in_K1 + (1-condition_res_1[1,:]) * (1-nu_in_K1),k))[-1]
                res_1 = func_gm1_res_1 * beta_Cm * cumprod_lambda_except_k
                # Normalization
                condition_res_0 = np.insert(condition, k+1, 0)
                func_gm1_res_0 = self.g_m(m=0, C_m= condition_res_0[0], C_m1=0, S_km=condition_res_0[1,:])
                res_0 = func_gm1_res_0 * beta_Cm * cumprod_lambda_except_k
                # Normalization
                prepera_to_sum_res_1.append(res_1)
                prepera_to_sum_res_0.append(res_0)
            sum_1 = np.sum(prepera_to_sum_res_1)
            sum_0 = np.sum(prepera_to_sum_res_0)
            self.bar_nu_h_to_s[k][0] = sum_1/(sum_1+sum_0)

        # calculate the rest m of nu_out
        condition_list_except_k = []
        for i in range(2**(K-1+2)):
            condition_str = ('{:0'+str(K-1+2)+'b}').format(i)
            condition_list = np.array([int(j) for j in condition_str])
            condition_list_except_k.append(condition_list)
        for m in np.array(range(M-1))+1:
            for k in range(K):
                prepera_to_sum_res_1 = []
                prepera_to_sum_res_0 = []
                for condition in condition_list_m1_except_k:
                    condition_res_1 = np.insert(condition, k+2, 1)
                    condition_res_1 = np.insert(condition, k+2, 0)
                    func_gm1_res_1 = self.g_m(m=m, C_m= condition_res_1[0], C_m1=condition_res_1[1], S_km=condition_res_1[2,:])
                    func_gm1_res_0 = self.g_m(m=m, C_m= condition_res_0[0], C_m1=condition_res_0[1], S_km=condition_res_0[2,:])
                    alpha_C_m1 = condition_res_1[1] * self.Forward_messages[m-1] + (1-condition_res_1[1]) * (1-self.Forward_messages[m-1])
                    beta_Cm = condition_res_1[0] * self.Backward_messages[m] + (1-condition_res_1[0]) * (1-self.Backward_messages[m])
                    nu_in_K1 = np.array([self.bar_nu_eta_to_s[k][m] for k in range(K)])
                    cumprod_lambda_except_k = np.cumprod(np.delete(condition_res_1[2,:] * nu_in_K1 + (1-condition_res_1[2,:]) * (1-nu_in_K1),k))[-1]
                    res_1 = func_gm1_res_1 * beta_Cm * cumprod_lambda_except_k
                    res_0 = func_gm1_res_0 * beta_Cm * cumprod_lambda_except_k
                    prepera_to_sum_res_1.append(res_1)
                    prepera_to_sum_res_0.append(res_0)
                sum_1 = np.sum(prepera_to_sum_res_1)
                sum_0 = np.sum(prepera_to_sum_res_0)
                self.bar_nu_h_to_s[k][m] = sum_1 / (sum_1 + sum_0)

        # bar_nu_h_to_s also equals to the <prior> probability of individual channel support s.
        pass

    def update_alpha(self):
        # first update alpha_1
        condition_K_list = []
        for i in range(2**K):
            condition_str = ('{:0'+str(K)+'b}').format(i)
            condition_list = np.array([int(j) for j in condition_str])
            condition_K_list.append(condition_list)
        
        prepare_to_sum_res_1 = []
        prepare_to_sum_res_0 = []
        for condition in condition_K_list:
            func_gm1_res_1 = self.g_m(m=0, C_m = 1, C_m1= 0, S_km=condition) 
            nu_in_K1 = np.array([self.bar_nu_eta_to_s[k][0] for k in range(K)])
            cumprod_lambda_K1 = np.cumprod(condition * nu_in_K1 + (1 - condition) * (1 - nu_in_K1))[-1]
            res_1 = func_gm1_res_1 * cumprod_lambda_K1
            func_gm1_res_0 = self.g_m(m=0, C_m = 0, C_m1= 0, S_km=condition) 
            res_0 = func_gm1_res_0 * cumprod_lambda_K1
            prepare_to_sum_res_1.append(res_1)
            prepare_to_sum_res_0.append(res_0)
        sum_1 = np.sum(prepare_to_sum_res_1)
        sum_0 = np.sum(prepare_to_sum_res_0)
        self.Forward_messages[0] = sum_1/(sum_1 + sum_0)

        # calculate the rest according to previos alpha
        condition_K_plus_C_m1 = []
        for i in range(2**(K+1)):
            condition_str = ('{:0'+str(K+1)+'b}').format(i)
            condition_list = np.array([int(j) for j in condition_str])
            condition_K_plus_C_m1.append(condition_list)
        
        for m in np.array(range(M-1)) + 1:
            prepare_to_sum_res_1 = []
            prepare_to_sum_res_0 = []
            for condition in condition_K_plus_C_m1:
                func_gm_res_1 = self.g_m(m=m, C_m= 1, C_m1= condition[0], S_km= condition[1,:])
                func_gm_res_0 = self.g_m(m=m, C_m= 0, C_m1= condition[0], S_km= condition[1,:])
                alpha_C_m1 = condition[0] * self.Forward_messages[m-1] + (1 - condition[0]) * (1 - self.Forward_messages[m-1])
                nu_in_Km = np.array([self.bar_nu_eta_to_s[k][m] for k in range(K)])
                cumprod_lambda_Km = np.cumprod(condition * nu_in_Km + (1 - condition) * (1- nu_in_Km) )[-1]
                res_1 = func_gm_res_1 * alpha_C_m1 * cumprod_lambda_Km
                res_0 = func_gm_res_0 * alpha_C_m1 * cumprod_lambda_Km
                prepare_to_sum_res_1.append(res_1)
                prepare_to_sum_res_0.append(res_0)
            sum_1 = np.sum(prepare_to_sum_res_1)
            sum_0 = np.sum(prepare_to_sum_res_0)
            self.Forward_messages[m] = sum_1/(sum_1+sum_0)
        pass

    def update_beta(self):
        # update the last beta
        self.Backward_messages[M-1] = 0.5
        # update the rest according to previous beta
        condition_K_plus_C_m = []
        for i in range(2**(K+1)):
            condition_str = ('{:0'+str(K+1)+'b}').format(i)
            condition_list = np.array([int(j) for j in condition_str])
            condition_K_plus_C_m.append(condition_list)
        
        for m in np.flipud(1+np.array(range(M-1))):
            prepare_to_sum_res_1 = []
            prepare_to_sum_res_0 = []
            for condition in condition_K_plus_C_m:
                func_gm_res_1 = self.g_m(m=m, C_m = condition[0], C_m1 = 1, S_km= condition[1:0])
                func_gm_res_0 = self.g_m(m=m, C_m = condition[0], C_m1 = 0, S_km= condition[1:0])
                beta_Cm = condition[0] * self.Backward_messages[m] +(1- condition[0]) * (1-self.Backward_messages[m])
                nu_in_Km = np.array([self.bar_nu_eta_to_s[k][m] for k in range(K)])
                cumprod_lambda_Km = np.cumprod(condition * nu_in_Km + (1 - condition) * (1- nu_in_Km) )[-1]
                res_1 = func_gm_res_1 * beta_Cm * cumprod_lambda_Km
                res_0 = func_gm_res_1 * beta_Cm * cumprod_lambda_Km
                prepare_to_sum_res_1.append(res_1)
                prepare_to_sum_res_0.append(res_0)
            sum_1 = np.sum(prepare_to_sum_res_1)
            sum_0 = np.sum(prepare_to_sum_res_0)
            self.Backward_messages[m-1] = sum_1/(sum_1+sum_0)
        pass

    def g_m(self, m=1, C_m = 0, C_m1 = 0, S_km = np.array([0,0,0,0])):
        # m cant be 0
        p01 = self.trans_pro_C['p01']
        p10 = self.trans_pro_C['p10']
        p1 = self.trans_pro_C['lambda']
        P_Skm_c_Cm = np.array([self.trans_pro_CS[k][m] for k in range(K)])
        p11 = 1 - p10
        if C_m == 0:
            return 0
        else:
            if m ==0 :
                P_C1 = C_m * p1 + (1-C_m) * (1 - p1)
                P_Skm_ConditionedOn_Cm = S_km * P_Skm_c_Cm + (1-S_km) * (1 - P_Skm_c_Cm)
                P_Skm_ConditionedOn_Cm_cumprod = np.cumprod(P_Skm_ConditionedOn_Cm)[-1]
                return P_C1 * P_Skm_ConditionedOn_Cm_cumprod
            else:
                P_cm1_cm = C_m1 * p11 + (1 - C_m1) * p01
                # P_cm = C_m * p1 + (1 - C_m) * (1 - p1)
                P_Skm_ConditionedOn_Cm = S_km * P_Skm_c_Cm + (1-S_km) * (1 - P_Skm_c_Cm)
                P_Skm_ConditionedOn_Cm_cumprod = np.cumprod(P_Skm_ConditionedOn_Cm)[-1]
                return P_cm1_cm * P_Skm_ConditionedOn_Cm_cumprod

    