import numpy as np
from scipy.integrate import solve_ivp

class SmartphoneBatteryModel:
    def __init__(self):
        # 电池参数
        self.Q0 = 4735.952          # 标称容量 (mAh)
        self.V_nom = 3.7          # 标称电压 (V)
        
        # 二阶RC等效电路参数
        self.R0 = 0.05            # 欧姆内阻 (Ohm) - 瞬态响应
        self.R1 = 0.03            # 电化学极化内阻 (Ohm)
        self.C1 = 2000.0          # 电化学极化电容 (F)
        self.R2 = 0.02            # 浓度极化内阻 (Ohm)
        self.C2 = 5000.0          # 浓度极化电容 (F)
        
        # 时间常数
        self.tau1 = self.R1 * self.C1  # ~60秒 (电化学极化)
        self.tau2 = self.R2 * self.C2  # ~100秒 (浓度极化)
        
        self.Ea = 35000.0         # 活化能 (J/mol)
        self.R_gas = 8.314        # 气体常数
        self.T_ref = 298.15       # 参考温度 (K)
        self.alpha = 0.5          # 传递系数
        self.F = 96485.0          # 法拉第常数
        
        # 热参数
        self.C_th = 200.0         # 热容 (J/K)
        self.h = 10.0             # 传热系数 (W/m2K)
        self.A = 0.01             # 表面积 (m2)
        
        # 老化参数
        self.k_aging = 5e-6       # 老化速率常数 (h^-1)
        
        # 组件功耗参数
        self.P_cpu_idle = 0.1     # CPU空闲功耗 (W)
        self.P_cpu_B=0.5             # CPU系数
        self.P_cpu_f=2             # CPU频率
        self.P_cpu_max = 3.0      # CPU最大功耗 (W)
        self.P_net_idle = 0.1     # 网络空闲功耗 (W)
        self.beta = 0.1           # 网络增量系数 (W/Mbps)
        self.P_gps = 0.1          # GPS功耗 (W)
        self.P_base = 0.05        # 基础功耗 (W)
        self.P_refresh= 60        #屏幕刷新率
        self.P_screen_square=1.14 #屏幕面积 dm2
        self.P_a=0.02                 #屏幕亮度系数
        self.P_screen_max = 0.6   # 屏幕最大功耗 (W)
        
    def V_oc(self, SOC):
        """开路电压曲线 (OCV-SOC关系)"""
        #SOC_safe = np.clip(SOC, 0.0, 1.0)
        return 3.39 + 2.65*SOC - 11.11*SOC**2 + 24.973*SOC**3
    
    def get_RC_params(self, SOC, T):
        """
        获取二阶RC参数（随SOC和温度变化）
        返回: R0, R1, R2 (考虑温度和SOC影响)
        """
        #SOC = np.clip(SOC, 0.05, 1.0)
        
        # 温度影响因子
        T_effect = np.exp(0.01*(1/T - 1/self.T_ref))
        
        # SOC影响因子（SOC越低，内阻越大）
        SOC_effect = 1 + 0.5 * (1 - SOC)
        
        # 各电阻随SOC和温度变化
        R0 = self.R0 * SOC_effect * T_effect
        R1 = self.R1 * SOC_effect * T_effect
        R2 = self.R2 * SOC_effect * T_effect
        
        return R0, R1, R2
    
    def f_T(self, T):
        """温度容量修正因子"""
        return np.exp(self.Ea/self.R_gas * (1/self.T_ref - 1/T))
    
    def f_aging(self, t):
        """老化容量修正因子"""
        return np.sqrt(1 - self.k_aging * t)
    
    def Q_eff(self, T, t):
        """有效容量"""
        return self.Q0 * self.f_T(T) * self.f_aging(t)
    
    def component_power(self, t, scenario):
        """计算各组件功耗"""
        power = self.P_base
        
        # 屏幕功耗
        if 'screen_on' in scenario:
            if scenario['screen_on'] ==True:
              brightness = scenario.get('brightness', 0.5)
              power += self.P_a *brightness*self.P_refresh*self.P_screen_square
        
        # CPU功耗
        if 'cpu_usage' in scenario:
            cpu_usage = scenario['cpu_usage']
            power += self.P_cpu_idle + cpu_usage*self.P_cpu_B*(self.P_cpu_f**3)
        
        # 网络功耗
        if 'data_rate' in scenario:
            data_rate = scenario['data_rate']
            power += self.P_net_idle + self.beta * data_rate
        
        # GPS功耗
        if 'gps_on' in scenario:
            if scenario['gps_on']==True:
               power += self.P_gps
        
        return power
    
    def solve_current(self, P_total, V_oc, U1, U2, R0):
        """
        考虑极化电压的电流求解
        终端电压: V_terminal = V_oc - I*R0 - U1 - U2
        功率: P = I * V_terminal
        求解: I^2*R0 - I*(V_oc - U1 - U2) + P = 0
        """
        # 有效开路电压（扣除极化电压）
        V_eff = V_oc - U1 - U2
        
        # 防护
        if V_eff <= 0.1 or P_total <= 0 or R0 <= 0:
            return 0.0
        
        # 求解二次方程: a*I^2 + b*I + c = 0
        a = R0
        b = -V_eff
        c = P_total
        
        delta = b**2 - 4*a*c
        
        if delta < 0:
            # 无实解，功率需求过大
            I = V_eff / (2 * R0)
        else:
            # 取较小的正根
            I = (-b - np.sqrt(delta)) / (2 * a)
        
        # 限制电流范围 (0-5A)
        I = np.clip(I, 0, 5.0)
        
        return I
    
    def model_equations(self, t, y, scenario_func):
        """
        改进的微分方程组（二阶RC模型）
        y = [SOC, T_batt, U1, U2]
        SOC: 荷电状态
        T_batt: 电池温度
        U1: 电化学极化电压
        U2: 浓度极化电压
        """
        SOC, T_batt, U1, U2 = y
        
        # 防止SOC过低
        if SOC <= 0:
            return [0, 0, 0, 0]
        
        # 获取当前场景参数
        scenario = scenario_func(t)
        
        # 计算总功耗
        P_total = self.component_power(t, scenario)
        
        # 计算开路电压
        V_oc = self.V_oc(SOC)
        
        # 获取RC参数（随SOC和温度变化）
        R0, R1, R2 = self.get_RC_params(SOC, T_batt)
        
        # 求解电流 (单位: A)
        I_A = self.solve_current(P_total, V_oc, U1, U2, R0)
        
        # 转换为mA用于SOC计算
        I_mA = I_A * 1000
        
        # 有效容量
        Q_eff = self.Q_eff(T_batt, t)
        
        # 1. SOC变化率 (1/s)
        dSOC_dt = -I_mA / (Q_eff * 3600)
        
        # 2. 电化学极化电压变化率 (V/s)
        # dU1/dt = (I*R1 - U1) / tau1
        dU1_dt = (I_A * R1 - U1) / self.tau1
        
        # 3. 浓度极化电压变化率 (V/s)
        # dU2/dt = (I*R2 - U2) / tau2
        dU2_dt = (I_A * R2 - U2) / self.tau2
        
        # 4. 温度变化率 (K/s)
        # 总热损耗 = I^2 * (R0 + R1 + R2)
        R_total = R0 + R1 + R2
        P_heat = I_A**2 * R_total
        P_cool = self.h * self.A * (T_batt - scenario.get('T_amb', 298.15))
        dT_dt = (P_heat - P_cool) / self.C_th
        
        return [dSOC_dt, dT_dt, dU1_dt, dU2_dt]
    
    def simulate(self, t_span, y0, scenario_func, max_step=60):
        """模拟电池放电"""
        sol = solve_ivp(
            lambda t, y: self.model_equations(t, y, scenario_func),
            t_span,
            y0,
            method='RK45',
            max_step=max_step,
            rtol=1e-6,
            atol=1e-9
        )
        return sol
    
    def find_empty_time(self, sol, V_cutoff=2.5, SOC_min=0.05):
        """
        找到电池放空时间（优化版）
        
        参数:
            sol: solve_ivp的求解结果
            V_cutoff: 截止电压 (V)，默认2.5V（较宽松）
            SOC_min: 最小SOC阈值，默认5%
        
        返回:
            放空时间 (s)，使用线性插值提高精度
        
        算法改进:
        1. 使用线性插值精确定位SOC越过阈值的时刻
        2. 同时检查终端电压是否低于截止电压
        3. 取两种条件中较早触发的时间
        """
        SOC = sol.y[0]
        T_batt = sol.y[1]
        U1 = sol.y[2]
        U2 = sol.y[3]
        t = sol.t
        
        # 方法1: 检查SOC是否低于阈值（带插值）
        idx_soc = np.where(SOC <= SOC_min)[0]
        t_soc_empty = None
        
        if len(idx_soc) > 0:
            i = idx_soc[0]
            if i > 0:
                # 线性插值找到精确的SOC=SOC_min时刻
                soc_before = SOC[i-1]
                soc_after = SOC[i]
                t_before = t[i-1]
                t_after = t[i]
                
                # 插值公式
                if abs(soc_after - soc_before) > 1e-10:
                    t_soc_empty = t_before + (SOC_min - soc_before) / (soc_after - soc_before) * (t_after - t_before)
                else:
                    t_soc_empty = t[i]
            else:
                t_soc_empty = t[i]
        
        # 方法2: 检查开路电压（简化版，不考虑负载）
        t_voltage_cutoff = None
        V_oc_array = np.array([self.V_oc(soc) for soc in SOC])
        
        idx_voltage = np.where(V_oc_array <= V_cutoff)[0]
        if len(idx_voltage) > 0:
            i = idx_voltage[0]
            if i > 0:
                # 线性插值
                v_before = V_oc_array[i-1]
                v_after = V_oc_array[i]
                t_before = t[i-1]
                t_after = t[i]
                
                if abs(v_after - v_before) > 1e-10:
                    t_voltage_cutoff = t_before + (V_cutoff - v_before) / (v_after - v_before) * (t_after - t_before)
                else:
                    t_voltage_cutoff = t[i]
            else:
                t_voltage_cutoff = t[i]
        
        # 取两种方法中较早的时间（更保守）
        empty_times = [t for t in [t_soc_empty, t_voltage_cutoff] if t is not None]
        
        if len(empty_times) > 0:
            return min(empty_times)
        else:
            # 未达到任何停止条件，返回最后时刻
            return t[-1]
    
    def get_terminal_voltage(self, SOC, T_batt, U1, U2, I_A):
        """
        计算终端电压
        V_terminal = V_oc - I*R0 - U1 - U2
        """
        V_oc = self.V_oc(SOC)
        R0, _, _ = self.get_RC_params(SOC, T_batt)
        V_terminal = V_oc - I_A * R0 - U1 - U2
        return V_terminal
