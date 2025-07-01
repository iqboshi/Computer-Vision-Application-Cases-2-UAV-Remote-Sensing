import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import NuSVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def Fun_Spectral_Correction(T, p, q, h, index, sort, m, n):
    ga = []
    for t in range(1, T + 1):
        if t <= p * T:
            a = 0
        else:
            t_q = (T - p * T) * q + p * T
            a = (-h / ((p * T - t_q) ** 2)) * ((t - t_q) ** 2) + h
            if a < 0:
                a = 0
        ga.append(a)

    ga = np.array(ga)
    fa = 1 + ga  # 偏移增强
    Fa_sorted = fa[np.argsort(sort)]  # sort is already sorted, this gives inverse mapping
    Fa = Fa_sorted[index]  # 映射回原图的 index 索引
    Fa = Fa.reshape(m, n)
    return Fa


def selectByGWOfun(X_train, y_train, X_test, y_test, SearchAgents_no, T, dim, feature_num, lb, ub):
    def fitness_function(mask):
        if np.sum(mask) == 0:
            return 0
        selected = np.where(mask == 1)[0]
        model = RandomForestRegressor(random_state=42)
        score = cross_val_score(model, X_train[:, selected], y_train, cv=3).mean()
        return score

    # 初始化群体
    positions = np.random.randint(0, 2, size=(SearchAgents_no, feature_num))
    Alpha_pos = np.zeros(feature_num)
    Alpha_score = -np.inf
    Beta_pos = np.zeros(feature_num)
    Beta_score = -np.inf
    Delta_pos = np.zeros(feature_num)
    Delta_score = -np.inf

    All_position = []
    All_score = []

    for t in range(T):
        for i in range(SearchAgents_no):
            mask = positions[i]
            score = fitness_function(mask)

            # 保存历史
            All_position.append(mask.copy())
            All_score.append(score)

            if score > Alpha_score:
                Delta_score, Delta_pos = Beta_score, Beta_pos.copy()
                Beta_score, Beta_pos = Alpha_score, Alpha_pos.copy()
                Alpha_score, Alpha_pos = score, mask.copy()
            elif score > Beta_score:
                Delta_score, Delta_pos = Beta_score, Beta_pos.copy()
                Beta_score, Beta_pos = score, mask.copy()
            elif score > Delta_score:
                Delta_score, Delta_pos = score, mask.copy()

        a = 2 - t * (2 / T)
        for i in range(SearchAgents_no):
            for j in range(feature_num):
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * Alpha_pos[j] - positions[i][j])
                X1 = Alpha_pos[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * Beta_pos[j] - positions[i][j])
                X2 = Beta_pos[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * Delta_pos[j] - positions[i][j])
                X3 = Delta_pos[j] - A3 * D_delta

                S = (X1 + X2 + X3) / 3
                positions[i][j] = 1 if S > 0.5 else 0

    return Alpha_pos, list(range(T)), Alpha_score, a, np.array(All_position), np.array(All_score)


def fun_init(A, X_tr, Y_tr, X_te):
    """
    参数：
        A     : ndarray，完整数据，shape = (样本数, 特征数+1)，最后一列为目标变量
        X_tr  : list/array，训练样本索引
        Y_tr  : list/array，选择的特征索引（从0开始）
        X_te  : list/array，测试样本索引

    返回：
        ps_output : 目标归一化器
        ps_input  : 特征归一化器
        p_train   : 归一化后的训练特征 (n_train, n_feat)
        p_test    : 归一化后的测试特征  (n_test, n_feat)
        t_train   : 归一化后的训练目标 (n_train,)
        t_test    : 归一化后的测试目标  (n_test,)
        T_train   : 原始训练目标
        T_test    : 原始测试目标
        P_train   : 原始训练特征
        P_test    : 原始测试特征
    """

    # 1. 特征提取
    P_train = A[X_tr][:, Y_tr]  # shape: (len(X_tr), len(Y_tr))
    T_train = A[X_tr][:, 23]    # 目标在第24列，对应索引 23
    P_test = A[X_te][:, Y_tr]
    T_test = A[X_te][:, 23]

    # 2. 归一化
    ps_input = MinMaxScaler(feature_range=(0, 1))
    p_train = ps_input.fit_transform(P_train)
    p_test = ps_input.transform(P_test)

    ps_output = MinMaxScaler(feature_range=(0, 1))
    t_train = ps_output.fit_transform(T_train.reshape(-1, 1)).ravel()
    t_test = ps_output.transform(T_test.reshape(-1, 1)).ravel()

    return ps_output, ps_input, p_train, p_test, t_train, t_test, T_train, T_test, P_train, P_test


def fun_model_results(p_train, t_train, p_test, t_test, T_train, T_test, M, N, c, g, ps_output):
    model = NuSVR(C=c, gamma=g, nu=0.5, kernel='rbf')  # nu=0.5 对应 -s 3
    model.fit(p_train, t_train)

    # 预测及反归一化
    t_pred = model.predict(p_test)
    T_pred = ps_output.inverse_transform(t_pred.reshape(-1, 1)).ravel()

    # 计算指标
    R1 = np.corrcoef(t_pred, t_test)[0, 1]
    rmse1 = np.sqrt(mean_squared_error(t_test, t_pred))
    MRE1 = np.mean(np.abs(t_test - t_pred) / (t_test + 1e-8))

    R2 = np.corrcoef(T_pred, T_test)[0, 1]
    rmse2 = np.sqrt(mean_squared_error(T_test, T_pred))
    MRE2 = np.mean(np.abs(T_test - T_pred) / (T_test + 1e-8))

    return R1, rmse1, MRE1, R2, rmse2, MRE2, model