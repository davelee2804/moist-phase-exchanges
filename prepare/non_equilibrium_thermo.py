import numpy as np

# Bowen and Thuburn, QJRMS, 2022
p_0d     = 1.0e+5
p_0sat   = 611.2
p_0v     = p_0sat
p_0vi    = p_0sat
#R_d      = 287.0
R_d      = 287.5 # LFRic
#R_v      = 461.0
R_v      = 461.51 # LFRic
T_0      = 273.15
alpha_0d = R_d*T_0/p_0d
alpha_0v = R_v*T_0/p_0v
alpha_l  = 0.001
alpha_i  = 0.0011
#c_pd     = 1004.0
c_pd     = 1005.0 # LFRic
c_pv     = 1885.0
c_vd     = c_pd - R_d
c_vv     = c_pv - R_v
c_i      = 2106.0
c_l      = 4186.0
L_0s     = 2.834e+6
L_0v     = 2.5e+6
L_0f     = L_0s - L_0v
L_00s    = L_0s - (c_pv - c_i)*T_0 + alpha_i*p_0v
L_00v    = L_0v - (c_pv - c_l)*T_0 + alpha_l*p_0v
L_00f    = L_00s - L_00v
c_vd_inv = 1.0/c_vd
c_vv_inv = 1.0/c_vv
c_l_inv  = 1.0/c_l
c_i_inv  = 1.0/c_i
eta_0    = 0.0

def u_d(rho,eta_d,q_d):
    return c_vd*T_0*np.exp((eta_d - c_pd)/c_vd)*np.power(alpha_0d*q_d*rho,R_d/c_vd)

def u_v(rho,eta_v,q_v):
    return c_vv*T_0*np.exp((eta_v - c_pv - L_00s/T_0)/c_vv)*np.power(alpha_0v*q_v*rho,R_v/c_vv) + L_00s

def u_l(eta_l):
    return c_l*T_0*np.exp((eta_l - (alpha_l*p_0sat + L_00f)/T_0)/c_l - 1.0) + L_00f

def u_i(eta_i):
    return c_i*T_0*np.exp((eta_i - alpha_i*p_0vi/T_0)/c_i - 1.0)

# T_d
def du_d_deta_d(rho,eta_d,q_d):
    return T_0*np.exp((eta_d - c_pd)/c_vd)*np.power(alpha_0d*q_d*rho,R_d/c_vd)

# T_v
def du_v_deta_v(rho,eta_v,q_v):
    return T_0*np.exp((eta_v - c_pv - L_00s/T_0)/c_vv)*np.power(alpha_0v*q_v*rho,R_v/c_vv)

# T_l
def du_l_deta_l(eta_l):
    return T_0*np.exp((eta_l - (alpha_l*p_0sat + L_00f)/T_0)/c_l - 1.0)

# T_i
def du_i_deta_i(eta_i):
    return T_0*np.exp((eta_i - alpha_i*p_0vi/T_0)/c_i - 1.0)

# p_d = q_d rho^2 du_d / drho
def p_d(rho,eta_d,q_d):
    return alpha_0d*q_d*q_d*rho*R_d*T_0*np.exp((eta_d - c_pd)/c_vd)*np.power(alpha_0d*q_d*rho,R_d/c_vd)
    #return alpha_0d*q_d*rho*R_d*T_0*np.exp((eta_d - c_pd)/c_vd)*np.power(alpha_0d*q_d*rho,R_d/c_vd)

# p_v = q_v rho^2 du_v / drho
def p_v(rho,eta_v,q_v):
    return alpha_0v*q_v*q_v*rho*R_v*T_0*np.exp((eta_v - c_pv - L_00s/T_0)/c_vv)*np.power(alpha_0v*q_v*rho,R_v/c_vv)
    #return alpha_0v*q_v*rho*R_v*T_0*np.exp((eta_v - c_pv - L_00s/T_0)/c_vv)*np.power(alpha_0v*q_v*rho,R_v/c_vv)

def chem_pots_via_gibbs(rho,eta,q_d,q_v,q_l,q_i):
    R   = q_d*R_d  + q_v*R_v
    c_v = q_d*c_vd + q_v*c_vv + q_l*c_l + q_i*c_i
    c_p = q_d*c_pd + q_v*c_pv + q_l*c_l + q_i*c_i

    gamma = c_p/c_v

    # Eldred et al, QJRMS, eqn 58
    T = T_0*np.exp((eta-eta_0)/c_p)*np.power(alpha_0d*q_d*rho,R_d*q_d/c_v)*np.power(alpha_0v*q_v*rho,R_v*q_v/c_v)
    # Eldred et al, QJRMS, eqn 57
    p = rho*R*T

    # Eldred et al, QJRMS, eqn 55 (mu_j = g_j/q_j)
    mu_v = T*(c_vv - c_vv*np.log(T/T_0) + R_v*np.log(q_v*R_v*p/(R*p_0v))) - c_vv*T_0 + L_0v + L_0f
    mu_l = T*(c_l  - c_l *np.log(T/T_0))                                  - c_l *T_0 + L_0f
    mu_i = T*(c_i  - c_i *np.log(T/T_0))                                  - c_i *T_0

    return mu_v, mu_l, mu_i

def calc_entropy(rho,theta,q_d,q_v,q_l,q_i):
    # Eldred et al., QJRMS (2022) eqn. 69
    c_p = q_d*c_pd + q_v*c_pv + q_l*c_l + q_i*c_i
    R   = R_d*q_d + R_v*q_v

    eta = c_p*np.log(theta/T_0) - R*np.log((p_0d+p_0v)/R) - R_d*q_d*np.log(R_d*q_d/p_0d) - R_v*q_v*np.log(R_v*q_v/p_0v) + eta_0

    return eta

def calc_temperature(rho,eta,q_d,q_v,q_l,q_i):
    # Eldred et al., QJRMS (2022) eqn. 58
    c_p = q_d*c_pd + q_v*c_pv + q_l*c_l + q_i*c_i
    c_v = q_d*c_vd + q_v*c_vv + q_l*c_l + q_i*c_i

    T = T_0*np.exp((eta-eta_0)/c_v)*np.power(alpha_0d*q_d*rho,R_d*q_d/c_v)*np.power(alpha_0v*q_v*rho,R_v*q_v/c_v)

    return T

def chem_pots(rho,eta,q_d,q_v,q_l,q_i):
    # Derivatives of the internal energy with respect to the 
    # mass fractions, with the internal energy given as:
    # Eldred et al., QJRMS (2022) eqn. 53
    c_v = q_d*c_vd + q_v*c_vv + q_l*c_l + q_i*c_i

    a = np.exp((eta - eta_0)/c_v)
    b = np.power(alpha_0d*q_d*rho,R_d*q_d/c_v)
    c = np.power(alpha_0v*q_v*rho,R_v*q_v/c_v)

    # scale by c_vj for phase j
    da_dq = a*(eta_0 - eta)/(c_v*c_v)

    # d(a^f(x))/dx = a^f(x) . log(a) . df(x)/dx
    # scale by c_vj for phase j
    db_dq = -b*np.log(alpha_0d*q_d*rho)*R_d*q_d/(c_v*c_v)

    # scale by c_vj for phase j = l,i
    dc_dq = -c*np.log(alpha_0v*q_v*rho)*R_v*q_v/(c_v*c_v)

    # d(a^f(x))/dx = a(x)^f(x) [ log(a) . df(x)/dx + f(x) da(x)/dx / a(x) ]
    dc_dqv = c*(np.log(alpha_0v*q_v*rho)*(c_v*R_v - c_vv*R_v*q_v)/(c_v*c_v) + R_v*q_v/(c_v*q_v))

    tmp1 = T_0*(a*b*c - 1.0)
    tmp2 = T_0*c_v*da_dq*b*c
    tmp3 = T_0*c_v*a*db_dq*c

    mu_v = c_vv*tmp1 + c_vv*(tmp2 + tmp3) + T_0*c_v*    a*b*dc_dqv - R_v*T_0 + (L_00v + L_00f)
    mu_l = c_l *tmp1 + c_l *(tmp2 + tmp3) + T_0*c_v*c_l*a*b*dc_dq  + L_00f
    mu_i = c_i *tmp1 + c_i *(tmp2 + tmp3) + T_0*c_v*c_i*a*b*dc_dq

    return mu_v, mu_l, mu_i

def chem_pots_2(rho,eta,q_d,q_v,q_l,q_i):
    # Derivatives of the internal energy with respect to the 
    # mass fractions, with the internal energy given as:
    # Eldred et al., QJRMS (2022) eqn. 53
    c_v = q_d * c_vd + q_v * c_vv + q_l * c_l + q_i * c_i
    c_v_inv = 1.0 / c_v

    _a = np.exp((eta - eta_0) * c_v_inv)
    _b = np.power(alpha_0d * q_d * rho, R_d * q_d * c_v_inv)
    _c = np.power(alpha_0v * q_v * rho, R_v * q_v * c_v_inv)
    T = T_0 * _a * _b * _c

    # d(a^f(x))/dx = a^f(x) . log(a) . df(x)/dx
    _ad = q_d * R_d * np.log(q_d * rho * alpha_0d)
    _av = q_v * R_v * np.log(q_v * rho * alpha_0v)
    _tmp = T - c_v_inv * T * (_ad + _av) - T_0

    #mu_v = c_vv * _tmp + T * R_v * np.log(q_v * rho * alpha_0v) + \
    #       c_v * T_0 * _a * _b * rho * alpha_0v * np.power(alpha_0v * q_v * rho, R_v * q_v * c_v_inv - 1.0) - \
    #       R_v * T_0 + L_00f + L_00s
    mu_v = c_vv * _tmp + T * R_v * np.log(q_v * rho * alpha_0v) + \
           T_0 * _a * _b * R_v * q_v * rho * alpha_0v * np.power(alpha_0v * q_v * rho, R_v * q_v * c_v_inv - 1.0) - \
           R_v * T_0 + L_00f + L_00v
    mu_l = c_l * _tmp + L_00f
    mu_i = c_i * _tmp

    return mu_v, mu_l, mu_i

def chem_pots_3(rho,eta,q_d,q_v,q_l,q_i):
    # Derivatives of the internal energy with respect to the 
    # mass fractions, with the internal energy given as:
    # Eldred et al., QJRMS (2022) eqn. 53
    c_v_inv = 1.0 / (q_d * c_vd + q_v * c_vv + q_l * c_l + q_i * c_i)

    _a = np.exp((eta - eta_0) * c_v_inv)
    _b = np.power(alpha_0d * q_d * rho, R_d * q_d * c_v_inv)
    _c = np.power(alpha_0v * q_v * rho, R_v * q_v * c_v_inv)
    T = T_0 * _a * _b * _c

    # d(a^f(x))/dx = a^f(x) . log(a) . df(x)/dx
    __d = R_d * q_d * np.log(alpha_0d * rho * q_d) * c_v_inv
    __v = R_v * q_v * np.log(alpha_0v * rho * q_v) * c_v_inv

    # d(a(x)^f(x))/dx = a(x)^f(x) . [ log(a) . df(x)/dx + f(x) / x ]

    #mu_v = c_vv * (T - T_0) - c_vv * T * __d * c_v_inv + \
    #       T * ((R_v - c_v_inv * R_v * c_vv * q_v) * np.log(alpha_0v * rho * q_v) + R_v) - \
    #       R_v * T_0 + L_00v + L_00f
    mu_v = (c_vv + R_v) * (T - T_0) - c_vv * T * __d + \
           T * (R_v - c_v_inv * R_v * c_vv * q_v) * np.log(alpha_0v * rho * q_v) + L_00v + L_00f
    mu_l = c_l * (T - T_0) - c_l * T * (__d + __v) + L_00f
    mu_i = c_i * (T - T_0) - c_i * T * (__d + __v)

    return mu_v, mu_l, mu_i

def chem_pots_eldred(rho, eta, q_d, q_v, q_l, q_i):
    # E6 - E8 of Eldred et al., QJRMS (2022)
    c_v = q_d * c_vd + q_v * c_vv + q_l * c_l + q_i * c_i
    c_v_inv = 1.0 / c_v
    eta_0v = 0.0
    eta_0l = 0.0
    eta_0i = 0.0

    T = T_0 * np.exp((eta - eta_0) * c_v_inv) \
            * np.power(alpha_0d * q_d * rho, R_d * q_d * c_v_inv) \
            * np.power(alpha_0v * q_v * rho, R_v * q_v * c_v_inv)

    _ad = q_d * R_d * np.log(q_d * rho * alpha_0d)
    _av = q_v * R_v * np.log(q_v * rho * alpha_0v)

    mu_v = T * (c_vv - c_vv * c_v_inv * (eta - eta_0 + _ad + _av) - eta_0v + \
           R_v * np.log(q_v * rho * alpah_0v) + R_v) - c_vv * T_0 + L_00v + L_00f - R_v * T_0
    mu_l = T * (c_l  - c_l  * c_v_inv * (eta - eta_0 + _ad + _av) - eta_0l) - c_l * T_0 + L_00f
    mu_i = T * (c_i  - c_i  * c_v_inv * (eta - eta_0 + _ad + _av) - eta_0i) - c_i * T_0 

    return mu_v, mu_l, mu_i

def calc_density(theta,exner,q_d,q_v,q_l,q_i):
    # Eldred et al., QJRMS (2022) eqn. 74
    c_p = q_d*c_pd + q_v*c_pv + q_l*c_l + q_i*c_i
    c_v = q_d*c_vd + q_v*c_vv + q_l*c_l + q_i*c_i
    R   = R_d*q_d + R_v*q_v

    # TODO: remove c_p scaling??
    rho = (p_0d/(R*theta))*np.power(exner/c_p, c_v/R)

    return rho

def calc_density_lfric(theta,exner,m_v):
    # From LFRic:
    # epsilon = Rd / Rv
    # \theta_vd = \theta * (1 + mr_v/\epsilon)
    # \Pi = (R_d\rho\theta_vd/p_0)^(\kappa/(1-\kappa))
    # rho = (p_zero*exner**((1.0_r_def - kappa)/kappa))/(rd*theta_vd)
    moist_dyn_gas = 1.0 + (R_v/R_d)*m_v
    rho = p_0d/(R_d*theta*moist_dyn_gas)*np.power(exner, c_vd/R_d)

    return rho

def calc_theta_moist_lfric(theta,m_v,m_cl,m_ci,m_r,m_s,m_g):
    # From LFRic:
    # moist_dyn_gas = 1.0 + mr_v / \epsilon
    # moist_dyn_tot = 1.0 + \sum_j m_j
    # theta_moist = theta * moist_dyn_gas / moist_dyn_tot
    moist_dyn_gas = 1.0 + (R_v/R_d)*m_v
    moist_dyn_tot = 1.0 + m_v + m_cl + m_ci + m_r + m_s + m_g
    theta_moist = theta*moist_dyn_gas/moist_dyn_tot
    
    return theta_moist

def internal_energy(rho,eta,q_d,q_v,q_l,q_i):
    # Eldred et al., QJRMS (2022) eqn. 53
    c_v = q_d*c_vd + q_v*c_vv + q_l*c_l + q_i*c_i

    a = np.exp((eta - eta_0)/c_v)
    b = np.power(alpha_0d*q_d*rho,R_d*q_d/c_v)
    c = np.power(alpha_0v*q_v*rho,R_v*q_v/c_v)

    return T_0*c_v*(a*b*c - 1.0) - q_v*R_v*T_0 + q_v*(L_00v + L_00f) + q_l*L_00f

def get_thermodynamic_quantities(density, entropy, qv, ql, qi):
    _Rd = 287.0
    _Rv = 461.0
    _cpd = 1004.0
    _cvd = _cpd - _Rd
    _cpv = 1885.0
    _cvv = _cpv - _Rv
    _cl = 4186.0
    _ci = 2106.0
    _logRd = np.log(_Rd)
    _T0 = 273.15
    _p0 = 611.2 # ??? 1.0e+5??
    _psat0 = 611.2
    _rho0 = _p0 / (_Rv * _T0)
    _logT0 = np.log(_T0)
    Lv0_ = 2.5e6
    Ls0_ = 2.834e6
    Lf0_ = Ls0_ - Lv0_
    _Lv0 = Lv0_ - (_cpv - _cl) * _T0
    _Ls0 = Ls0_ - (_cpv - _ci) * _T0
    _Lf0 = _Ls0 - _Lv0
    _c0 = _cpv + (_Ls0 / _T0) - _cvv * _logT0 + _Rv * np.log(_rho0)
    _c1 = _cl + (_Lf0 / _T0) - _cl * _logT0
    _c2 = _ci - _ci * _logT0

    qw = qv + ql + qi
    qd = 1 - qw

    R = qv * _Rv + qd * _Rd
    cv = qd * _cvd + qv * _cvv + ql * _cl + qi * _ci

    logqv = np.log(qv)
    logqd = np.log(qd)
    logdensity = np.log(density)

    cvlogT = entropy + R * logdensity + qd * _Rd * (logqd + _logRd) + qv * _Rv * logqv
    cvlogT += -qv * _c0 - ql * _c1 - qi * _c2
    logT = (1 / cv) * cvlogT
    T = np.exp(logT)

    p = density * R * T

    specific_ie = cv * T + qv * _Ls0 + ql * _Lf0
    enthalpy = specific_ie + p / density
    ie = density * specific_ie

    dlogTdqv = (1 / cv) * (_Rv * logdensity + _Rv * logqv + _Rv - _c0)
    dlogTdqv += -(1 / cv) * logT * _cvv
    dTdqv = dlogTdqv * T

    dlogTdql = (1 / cv) * (-_c1)
    dlogTdql += -(1 / cv) * logT * _cl
    dTdql = dlogTdql * T

    dlogTdqi = (1 / cv) * (-_c2)
    dlogTdqi += -(1 / cv) * logT * _ci
    dTdqi = dlogTdqi * T

    dlogTdqd = (1 / cv) * (_Rd * logdensity + _Rd * (logqd + np.log(_Rd)) + _Rd)
    dlogTdqd += -(1 / cv) * logT * _cvd
    dTdqd = T * dlogTdqd

    # these are just the Gibbs functions
    mu_d = cv * dTdqd + _cvd * T
    mu_v = cv * dTdqv + _cvv * T + _Ls0
    mu_l = cv * dTdql + _cl * T + _Lf0
    mu_i = cv * dTdqi + _ci * T

    return mu_v, mu_l, mu_i, T, mu_d

def eta_d_to_eta_k(rho, eta, q_d, q_v, q_l, q_i):
    corr = c_vd*q_d*np.log(T_0) + R_d*q_d*np.log(T_0/p_0d) + q_v*(c_pv + L_00s/T_0) + q_l*(c_l + L_00f/T_0) + q_i*c_i

    return eta + corr
