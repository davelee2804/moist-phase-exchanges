import numpy as np
import netCDF4
from non_equilibrium_thermo import *

'''
Parse the lfric data and convert to variables necessary to train against

inputs:
    mixing ratios (vapor, liquid cloud, ice cloud, rain, graupel, snow)
    velocities (3d)
    exner pressure
    potential temperature

outputs:
    Dq_v/Dt  : material transport of water vapor phase
    Dq_l/Dt  : material transport of water liquid phase
    Dq_i/Dt  : material transport of water ice phase
    mu_v     : chemical potential of water vapor phase (scaled by the density)
    mu_l     : chemical potential of water liquid phase (scaled by the density)
    mu_i     : chemical potential of water ice phase (scaled by the density)
    temp     : temperature (thermal equilibrium, same for all phases)
    rho      : density (mechanical equilibrium, same for all phases)
'''

# get the vertical level index above which the levels no longer follow the orography
def eta_c_index(levels,domain_height,stretching_height):
    eta_c = stretching_height/domain_height
    for ii in np.arange(len(levels)):
        if levels[ii] > eta_c:
            return ii - 1

n_horiz = 480
left = 120#40
right = n_horiz - left
#bot_level_index = 23 # no liquid clouds below approx 2km altitude
bot_level_index = 2 # no liquid clouds below approx 2km altitude
top_level_index = 70 # (90 levels) # 63 
#dtInv = 1.0/120.0
dtInv = 1.0/60.0

#filename = '/g/data/dp9/dl9118/lfric_ral_training/20210105T0000Z/480x480/lfric/lfric_ral_model_levels_006_012.nc'
#orogfilename = '/g/data/access/projects/access/lfric/data/ancils/proto-rns/darwin_480x480/orography/globe30/qrparm.orog.ugrid.nc'
#filename = '/scratch/dp9/dl9118/cylc-run/rns_moist-phase-training/run3/share/cycle/20250119T0000Z/RMED-test/DAR/RAL4beta1p1/lfric/lfric_ral_model_levels_006_012.nc'
#orogfilename = '/scratch/dp9/dl9118/cylc-run/ras_hot_080926/run13/share/data/ancils/DAR/orography/SRTM_3arc/v3/qrparm.orog_mean.nc'
filename = '/g/data/dp9/dl9118/lfric_ral_training/20210105T0000Z/480x480/lfric/lfric_ral_model_levels_006_012_90levs.nc'
orogfilename = '/g/data/dp9/dl9118/lfric_ral_training/20210105T0000Z/480x480/lfric/qrparm.orog_mean_90levs.nc'

#full_levels = np.array([0.0000000E+00,   0.1250000E-03,   0.5416666E-03,   0.1125000E-02,   0.1875000E-02,\
#  0.2791667E-02,   0.3875000E-02,   0.5125000E-02,   0.6541667E-02,   0.8125000E-02,\
#  0.9875000E-02,   0.1179167E-01,   0.1387500E-01,   0.1612500E-01,   0.1854167E-01,\
#  0.2112500E-01,   0.2387500E-01,   0.2679167E-01,   0.2987500E-01,   0.3312500E-01,\
#  0.3654167E-01,   0.4012500E-01,   0.4387500E-01,   0.4779167E-01,   0.5187500E-01,\
#  0.5612501E-01,   0.6054167E-01,   0.6512500E-01,   0.6987500E-01,   0.7479167E-01,\
#  0.7987500E-01,   0.8512500E-01,   0.9054167E-01,   0.9612500E-01,   0.1018750E+00,\
#  0.1077917E+00,   0.1138750E+00,   0.1201250E+00,   0.1265417E+00,   0.1331250E+00,\
#  0.1398750E+00,   0.1467917E+00,   0.1538752E+00,   0.1611287E+00,   0.1685623E+00,\
#  0.1761954E+00,   0.1840590E+00,   0.1921980E+00,   0.2006732E+00,   0.2095645E+00,\
#  0.2189729E+00,   0.2290236E+00,   0.2398690E+00,   0.2516917E+00,   0.2647077E+00,\
#  0.2791699E+00,   0.2953717E+00,   0.3136506E+00,   0.3343919E+00,   0.3580330E+00,\
#  0.3850676E+00,   0.4160496E+00,   0.4515977E+00,   0.4924007E+00,   0.5392213E+00,\
#  0.5929016E+00,   0.6543679E+00,   0.7246365E+00,   0.8048183E+00,   0.8961251E+00,  0.1000000E+01])

#half_levels = np.array([0.6249999E-04,   0.3333333E-03,   0.8333333E-03,   0.1500000E-02,   0.2333333E-02,\
#  0.3333333E-02,   0.4500000E-02,   0.5833333E-02,   0.7333333E-02,   0.9000000E-02,\
#  0.1083333E-01,   0.1283333E-01,   0.1500000E-01,   0.1733333E-01,   0.1983333E-01,\
#  0.2250000E-01,   0.2533333E-01,   0.2833333E-01,   0.3150000E-01,   0.3483333E-01,\
#  0.3833333E-01,   0.4200000E-01,   0.4583333E-01,   0.4983333E-01,   0.5400000E-01,\
#  0.5833334E-01,   0.6283334E-01,   0.6750000E-01,   0.7233334E-01,   0.7733333E-01,\
#  0.8250000E-01,   0.8783333E-01,   0.9333333E-01,   0.9900000E-01,   0.1048333E+00,\
#  0.1108333E+00,   0.1170000E+00,   0.1233333E+00,   0.1298333E+00,   0.1365000E+00,\
#  0.1433333E+00,   0.1503334E+00,   0.1575020E+00,   0.1648455E+00,   0.1723789E+00,\
#  0.1801272E+00,   0.1881285E+00,   0.1964356E+00,   0.2051189E+00,   0.2142687E+00,\
#  0.2239982E+00,   0.2344463E+00,   0.2457803E+00,   0.2581997E+00,   0.2719388E+00,\
#  0.2872708E+00,   0.3045112E+00,   0.3240212E+00,   0.3462124E+00,   0.3715503E+00,\
#  0.4005586E+00,   0.4338236E+00,   0.4719992E+00,   0.5158110E+00,   0.5660614E+00,\
#  0.6236348E+00,   0.6895022E+00,   0.7647274E+00,   0.8504717E+00,   0.9480625E+00])

full_levels = np.array([0.0000000E+00,   0.1250000E-03,   0.4285715E-03,   0.9107144E-03,   0.1571428E-02,\
   0.2410714E-02,   0.3428571E-02,   0.4625000E-02,   0.6000000E-02,   0.7553572E-02,\
   0.9285714E-02,   0.1119643E-01,   0.1328572E-01,   0.1555357E-01,   0.1800000E-01,\
   0.2062500E-01,   0.2342857E-01,   0.2641072E-01,   0.2957143E-01,   0.3291072E-01,\
   0.3642858E-01,   0.4012500E-01,   0.4400000E-01,   0.4805358E-01,   0.5228572E-01,\
   0.5669644E-01,   0.6128573E-01,   0.6605358E-01,   0.7100001E-01,   0.7612501E-01,\
   0.8142858E-01,   0.8691073E-01,   0.9257144E-01,   0.9841072E-01,   0.1044286E+00,\
   0.1106250E+00,   0.1170000E+00,   0.1235536E+00,   0.1302857E+00,   0.1371964E+00,\
   0.1442857E+00,   0.1515690E+00,   0.1590695E+00,   0.1667955E+00,   0.1747552E+00,\
   0.1829564E+00,   0.1914075E+00,   0.2001169E+00,   0.2090935E+00,   0.2183466E+00,\
   0.2278855E+00,   0.2377203E+00,   0.2478612E+00,   0.2583188E+00,   0.2691042E+00,\
   0.2802286E+00,   0.2917038E+00,   0.3035420E+00,   0.3157557E+00,   0.3283577E+00,\
   0.3413614E+00,   0.3547803E+00,   0.3686285E+00,   0.3829205E+00,   0.3976710E+00,\
   0.4128953E+00,   0.4286089E+00,   0.4448278E+00,   0.4615685E+00,   0.4788476E+00,\
   0.4966824E+00,   0.5150903E+00,   0.5340893E+00,   0.5536979E+00,   0.5739346E+00,\
   0.5948187E+00,   0.6163697E+00,   0.6386074E+00,   0.6615524E+00,   0.6852252E+00,\
   0.7096471E+00,   0.7348395E+00,   0.7608244E+00,   0.7876242E+00,   0.8152616E+00,\
   0.8437597E+00,   0.8731421E+00,   0.9034327E+00,   0.9346560E+00,   0.9668368E+00,0.1000000E+01])

half_levels = np.array([0.6250001E-04,   0.2767857E-03,   0.6696429E-03,   0.1241071E-02,   0.1991071E-02,\
   0.2919643E-02,   0.4026786E-02,   0.5312500E-02,   0.6776786E-02,   0.8419643E-02,\
   0.1024107E-01,   0.1224107E-01,   0.1441964E-01,   0.1677679E-01,   0.1931250E-01,\
   0.2202679E-01,   0.2491965E-01,   0.2799107E-01,   0.3124107E-01,   0.3466965E-01,\
   0.3827679E-01,   0.4206250E-01,   0.4602679E-01,   0.5016965E-01,   0.5449108E-01,\
   0.5899108E-01,   0.6366965E-01,   0.6852679E-01,   0.7356251E-01,   0.7877679E-01,\
   0.8416965E-01,   0.8974108E-01,   0.9549108E-01,   0.1014196E+00,   0.1075268E+00,\
   0.1138125E+00,   0.1202768E+00,   0.1269197E+00,   0.1337411E+00,   0.1407411E+00,\
   0.1479274E+00,   0.1553192E+00,   0.1629325E+00,   0.1707753E+00,   0.1788558E+00,\
   0.1871819E+00,   0.1957622E+00,   0.2046052E+00,   0.2137200E+00,   0.2231160E+00,\
   0.2328029E+00,   0.2427908E+00,   0.2530900E+00,   0.2637115E+00,   0.2746664E+00,\
   0.2859662E+00,   0.2976229E+00,   0.3096489E+00,   0.3220567E+00,   0.3348595E+00,\
   0.3480708E+00,   0.3617044E+00,   0.3757745E+00,   0.3902957E+00,   0.4052831E+00,\
   0.4207521E+00,   0.4367184E+00,   0.4531982E+00,   0.4702081E+00,   0.4877650E+00,\
   0.5058863E+00,   0.5245898E+00,   0.5438936E+00,   0.5638163E+00,   0.5843767E+00,\
   0.6055942E+00,   0.6274885E+00,   0.6500799E+00,   0.6733888E+00,   0.6974362E+00,\
   0.7222433E+00,   0.7478320E+00,   0.7742243E+00,   0.8014429E+00,   0.8295106E+00,\
   0.8584509E+00,   0.8882874E+00,   0.9190444E+00,   0.9507464E+00,   0.9834184E+00])

stretching_height = 16641.984
domain_height = 40000.0
eta_c_inv = domain_height/stretching_height
i_eta_c = eta_c_index(full_levels,domain_height,stretching_height)
print('eta_c index: ' + str(eta_c_index))
eta_levels = full_levels[bot_level_index:top_level_index]
i_eta_c = i_eta_c - bot_level_index
z_levels = np.zeros(eta_levels.shape)
print('eta_c index: ' + str(eta_c_index))

# map from eta coordinate to z coordinate for a given orographic surface height
def eta2z_smooth(levels,eta_levels,eta_c_inv,i_eta_c,surface_height,domain_height):
    tmp = 1.0 - eta_c_inv*eta_levels[:i_eta_c]
    levels[:] = domain_height*eta_levels
    levels[:i_eta_c] = levels[:i_eta_c] + surface_height*tmp*tmp
    return levels

def get_and_reshape_field(fieldname,n_horiz,ncfile):
    print('loading field: ' + fieldname)
    field = ncfile.variables[fieldname][:]
    n_time = field.shape[0]
    n_levs = field.shape[1]
    field = field.reshape((n_time,n_levs,n_horiz,n_horiz))
    field = field[:,bot_level_index:top_level_index,left:right,left:right]
    field = np.flip(field,axis=2)
    return field

def horiz_gradients(field,cos_theta_inv,dlon,dlat):
    dfdx = np.gradient(field,dlon,axis=1)*cos_theta_inv
    dfdy = np.gradient(field,dlat,axis=0)
    return dfdx, dfdy

def vert_gradient(q_v,q_l,q_i,eta,levels,hgt,hgt_c_inv,i_hgt_c,surface_height,domain_height):
    z = eta2z_smooth(levels,hgt,hgt_c_inv,i_hgt_c,surface_height,domain_height)
    dqv_dz = np.gradient(q_v,z)
    dql_dz = np.gradient(q_l,z)
    dqi_dz = np.gradient(q_i,z)
    deta_dz = np.gradient(eta,z)
    return dqv_dz, dql_dz, dqi_dz, deta_dz

orogfile = netCDF4.Dataset(orogfilename, 'r')
#orog = orogfile.variables['surface_altitude'][:]
orog = orogfile.variables['orography_mean_height'][:]
orog = orog.reshape((n_horiz,n_horiz))
orog = np.flip(orog,axis=0)
orog = orog[left:right,left:right]

datafile = netCDF4.Dataset(filename)

face_x = datafile.variables["Mesh2d_face_x"][:]
face_x = face_x.reshape((n_horiz,n_horiz))
face_x = np.flip(face_x,axis=0)
face_x = face_x[left:right,left:right]

face_y = datafile.variables["Mesh2d_face_y"][:]
face_y = face_y.reshape((n_horiz,n_horiz))
face_y = np.flip(face_y,axis=0)
face_y = face_y[left:right,left:right]

# dimensions are organised as: [latitude,longitude]
rad_earth = 6371220.0 # radius of the earth, m
meters_per_degree = 2.0*np.pi*rad_earth/360.0
longitude = face_x[0,:]
latitude  = face_y[:,0]
dlon = meters_per_degree*(longitude[1]-longitude[0])
dlat = meters_per_degree*(latitude[1]-latitude[0])
cos_theta_inv = 1.0/np.cos(np.deg2rad(face_y))

theta = get_and_reshape_field('theta',n_horiz,datafile)
exner = get_and_reshape_field('exner_in_wth',n_horiz,datafile)
u = get_and_reshape_field('u_in_w3',n_horiz,datafile)
v = get_and_reshape_field('v_in_w3',n_horiz,datafile)
w = get_and_reshape_field('w_in_wth',n_horiz,datafile)
m_v = get_and_reshape_field('m_v',n_horiz,datafile)
m_cl = get_and_reshape_field('m_cl',n_horiz,datafile)
m_r = get_and_reshape_field('m_r',n_horiz,datafile)
m_ci = get_and_reshape_field('m_ci',n_horiz,datafile)
m_s = get_and_reshape_field('m_s',n_horiz,datafile)
m_g = get_and_reshape_field('m_g',n_horiz,datafile)

Dq_v = np.zeros(m_v[0,:,:,:].shape)
Dq_l = np.zeros(m_v[0,:,:,:].shape)
Dq_i = np.zeros(m_v[0,:,:,:].shape)
Deta = np.zeros(m_v[0,:,:,:].shape)

dqv_dx = np.zeros(m_v[0,:,:,:].shape)
dql_dx = np.zeros(m_v[0,:,:,:].shape)
dqi_dx = np.zeros(m_v[0,:,:,:].shape)
dqv_dy = np.zeros(m_v[0,:,:,:].shape)
dql_dy = np.zeros(m_v[0,:,:,:].shape)
dqi_dy = np.zeros(m_v[0,:,:,:].shape)
dqv_dz = np.zeros(m_v[0,:,:,:].shape)
dql_dz = np.zeros(m_v[0,:,:,:].shape)
dqi_dz = np.zeros(m_v[0,:,:,:].shape)
deta_dx = np.zeros(m_v[0,:,:,:].shape)
deta_dy = np.zeros(m_v[0,:,:,:].shape)
deta_dz = np.zeros(m_v[0,:,:,:].shape)

for time in np.arange(360-2)+1:
    # compute the mass fractions
    print(str(time)+'\tcomputing the mass fractions...')

    # LFRic: see iau_humidity_inc for these conversions
    tmp = 1.0/(1.0 + m_v[time-1,:,:,:] + m_cl[time-1,:,:,:] + m_ci[time-1,:,:,:] + m_r[time-1,:,:,:] + m_s[time-1,:,:,:] + m_g[time-1,:,:,:])
    q_v_tm1 = m_v[time-1,:,:,:]*tmp
    q_l_tm1 = (m_cl[time-1,:,:,:] + m_r[time-1,:,:,:])*tmp
    q_i_tm1 = (m_ci[time-1,:,:,:] + m_s[time-1,:,:,:] + m_g[time-1,:,:,:])*tmp

    tmp = 1.0/(1.0 + m_v[time,:,:,:] + m_cl[time,:,:,:] + m_ci[time,:,:,:] + m_r[time,:,:,:] + m_s[time,:,:,:] + m_g[time,:,:,:])
    q_v_t = m_v[time,:,:,:]*tmp
    q_l_t = (m_cl[time,:,:,:] + m_r[time,:,:,:])*tmp
    q_i_t = (m_ci[time,:,:,:] + m_s[time,:,:,:] + m_g[time,:,:,:])*tmp

    tmp = 1.0/(1.0 + m_v[time+1,:,:,:] + m_cl[time+1,:,:,:] + m_ci[time+1,:,:,:] + m_r[time+1,:,:,:] + m_s[time+1,:,:,:] + m_g[time+1,:,:,:])
    q_v_tp1 = m_v[time+1,:,:,:]*tmp
    q_l_tp1 = (m_cl[time+1,:,:,:] + m_r[time+1,:,:,:])*tmp
    q_i_tp1 = (m_ci[time+1,:,:,:] + m_s[time+1,:,:,:] + m_g[time+1,:,:,:])*tmp

    q_d_t = 1.0 - q_v_t - q_l_t - q_i_t
    q_d_tm1 = 1.0 - q_v_tm1 - q_l_tm1 - q_i_tm1
    q_d_tp1 = 1.0 - q_v_tp1 - q_l_tp1 - q_i_tp1

    # compute the thermodynamic state for the extrinsic variables
    print(str(time)+'\tcomputing the density...')
    #rho = calc_density(theta[time,:,:,:],exner[time,:,:,:],q_d_t,q_v_t,q_l_t,q_i_t)
    rho = calc_density_lfric(theta[time,:,:,:],exner[time,:,:,:],m_v[time,:,:,:])
    rho_tm1 = calc_density_lfric(theta[time-1,:,:,:],exner[time-1,:,:,:],m_v[time-1,:,:,:])
    rho_tp1 = calc_density_lfric(theta[time+1,:,:,:],exner[time+1,:,:,:],m_v[time+1,:,:,:])
    print(f'\t\trho min/max:\t{np.min(rho.flatten()):.4e}\t{np.max(rho.flatten()):.4e}')

    print(str(time)+'\tcomputing the entropy...')
    #theta_moist = calc_theta_moist_lfric(theta[time,:,:,:],m_v[time,:,:,:],m_cl[time,:,:,:],m_ci[time,:,:,:],m_r[time,:,:,:],m_s[time,:,:,:],m_g[time,:,:,:])
    #theta_moist_tm1 = calc_theta_moist_lfric(theta[time-1,:,:,:],m_v[time-1,:,:,:],m_cl[time-1,:,:,:],m_ci[time-1,:,:,:],m_r[time-1,:,:,:],m_s[time-1,:,:,:],m_g[time-1,:,:,:])
    #theta_moist_tp1 = calc_theta_moist_lfric(theta[time+1,:,:,:],m_v[time+1,:,:,:],m_cl[time+1,:,:,:],m_ci[time+1,:,:,:],m_r[time+1,:,:,:],m_s[time+1,:,:,:],m_g[time+1,:,:,:])
    #print(f'\t\ttheta min/max:\t{np.min(theta_moist.flatten()):.4e}\t{np.max(theta_moist.flatten()):.4e}')
    print(f'\t\ttheta min/max:\t{np.min(theta.flatten()):.4e}\t{np.max(theta.flatten()):.4e}')
    eta = calc_entropy(rho,theta[time,:,:,:],q_d_t,q_v_t,q_l_t,q_i_t)
    eta_m1 = calc_entropy(rho_tm1,theta[time-1,:,:,:],q_d_tm1,q_v_tm1,q_l_tm1,q_i_tm1)
    eta_p1 = calc_entropy(rho_tp1,theta[time+1,:,:,:],q_d_tp1,q_v_tp1,q_l_tp1,q_i_tp1)
    eta = eta_d_to_eta_k(rho, eta, q_d_t, q_v_t, q_l_t, q_i_t)
    eta_m1 = eta_d_to_eta_k(rho_tm1, eta_m1, q_d_tm1, q_v_tm1, q_l_tm1, q_i_tm1)
    eta_p1 = eta_d_to_eta_k(rho_tp1, eta_p1, q_d_tp1, q_v_tp1, q_l_tp1, q_i_tp1)
    #eta = calc_entropy(rho,theta_moist,q_d_t,q_v_t,q_l_t,q_i_t)
    #eta_m1 = calc_entropy(rho_tm1,theta_moist_tm1,q_d_tm1,q_v_tm1,q_l_tm1,q_i_tm1)
    #eta_p1 = calc_entropy(rho_tp1,theta_moist_tp1,q_d_tp1,q_v_tp1,q_l_tp1,q_i_tp1)
    print(f'\t\teta min/max:\t{np.min(eta.flatten()):.4e}\t{np.max(eta.flatten()):.4e}')

    print(str(time)+'\tcomputing the chemical potentials...')
    #mu_v, mu_l, mu_i = chem_pots_3(rho,eta,q_d_t,q_v_t,q_l_t,q_i_t)
    mu_v, mu_l, mu_i, temp, mu_d = get_thermodynamic_quantities(rho, eta, q_v_t, q_l_t, q_i_t)

    # compute the loss functions

    # time derivatives
    Dq_v[:,:,:] = 0.5*dtInv*(q_v_tp1 - q_v_tm1)
    Dq_l[:,:,:] = 0.5*dtInv*(q_l_tp1 - q_l_tm1)
    Dq_i[:,:,:] = 0.5*dtInv*(q_i_tp1 - q_i_tm1)
    Deta[:,:,:] = 0.5*dtInv*(eta_p1 - eta_m1)

    # vertical gradients
    for ii in np.arange(orog.shape[0]):
        for jj in np.arange(orog.shape[1]):
            surface_height = orog[ii,jj];
            _dqv_dz, _dql_dz, _dqi_dz, _deta_dz = vert_gradient(q_v_t[:,ii,jj],q_l_t[:,ii,jj],q_i_t[:,ii,jj],eta[:,ii,jj],\
                    z_levels,eta_levels,eta_c_inv,i_eta_c,surface_height,domain_height)
            dqv_dz[:,ii,jj] = w[time,:,ii,jj]*_dqv_dz
            dql_dz[:,ii,jj] = w[time,:,ii,jj]*_dql_dz
            dqi_dz[:,ii,jj] = w[time,:,ii,jj]*_dqi_dz
            deta_dz[:,ii,jj] = w[time,:,ii,jj]*_deta_dz

    Dq_v[:,:,:] = Dq_v[:,:,:] + dqv_dz[:,:,:]
    Dq_l[:,:,:] = Dq_l[:,:,:] + dql_dz[:,:,:]
    Dq_i[:,:,:] = Dq_i[:,:,:] + dqi_dz[:,:,:]
    Deta[:,:,:] = Deta[:,:,:] + deta_dz[:,:,:]

    # horizontal gradients
    for kk in np.arange(Dq_v.shape[0]):
        # velocity is on the half levels, need to average onto full levels
        if kk == 0:
            _u = u[time,kk,:,:]
            _v = v[time,kk,:,:]
        elif kk == Dq_v.shape[0]-1:
            _u = u[time,kk,:,:]
            _v = v[time,kk,:,:]
        else:
            _u = 0.5*(u[time,kk-1,:,:] + u[time,kk,:,:])
            _v = 0.5*(v[time,kk-1,:,:] + v[time,kk,:,:])

        _dqv_dx, _dqv_dy = horiz_gradients(q_v_t[kk,:,:],cos_theta_inv,dlon,dlat)
        dqv_dx[kk,:,:] = _u*_dqv_dx
        dqv_dy[kk,:,:] = _v*_dqv_dy

        _dql_dx, _dql_dy = horiz_gradients(q_l_t[kk,:,:],cos_theta_inv,dlon,dlat)
        dql_dx[kk,:,:] = _u*_dql_dx
        dql_dy[kk,:,:] = _v*_dql_dy

        _dqi_dx, _dqi_dy = horiz_gradients(q_i_t[kk,:,:],cos_theta_inv,dlon,dlat)
        dqi_dx[kk,:,:] = _u*_dqi_dx
        dqi_dy[kk,:,:] = _v*_dqi_dy

        _deta_dx, _deta_dy = horiz_gradients(eta[kk,:,:],cos_theta_inv,dlon,dlat)
        deta_dx[kk,:,:] = _u*_deta_dx
        deta_dy[kk,:,:] = _v*_deta_dy

    Dq_v[:,:,:] = Dq_v[:,:,:] + dqv_dx[:,:,:] + dqv_dy[:,:,:]
    Dq_l[:,:,:] = Dq_l[:,:,:] + dql_dx[:,:,:] + dql_dy[:,:,:]
    Dq_i[:,:,:] = Dq_i[:,:,:] + dqi_dx[:,:,:] + dqi_dy[:,:,:]
    Deta[:,:,:] = Deta[:,:,:] + deta_dx[:,:,:] + deta_dy[:,:,:]

    use_l = np.abs(Dq_l.flatten()) > 1.0e-9
    use_v = np.abs(Dq_v.flatten()) > 1.0e-9
    use_i = np.abs(Dq_i.flatten()) > 1.0e-9
    use = np.logical_or(use_v,use_i)
    use = np.logical_or(use,use_l)
    print('number of entries to add: ' + str(np.sum(use)))

    # filter the values where the change in liquid water is too small
    Dq_v_use = Dq_v.flatten()[use]
    Dq_l_use = Dq_l.flatten()[use]
    Dq_i_use = Dq_i.flatten()[use]
    mu_v_use = mu_v.flatten()[use]
    mu_l_use = mu_l.flatten()[use]
    mu_i_use = mu_i.flatten()[use]
    mu_d_use = mu_d.flatten()[use]
    rho_use  = rho.flatten()[use]
    temp_use = temp.flatten()[use]
    eta_use  = eta.flatten()[use]
    Deta_use = Deta.flatten()[use]
    qd_use = q_d_t.flatten()[use]
    qv_use = q_v_t.flatten()[use]
    ql_use = q_l_t.flatten()[use]
    qi_use = q_i_t.flatten()[use]

    # save the loss function components to file at the time step
    with open('Dq_v_loss.txt','a') as f:
        np.savetxt(f, Dq_v_use)
    with open('Dq_l_loss.txt','a') as f:
        np.savetxt(f, Dq_l_use)
    with open('Dq_i_loss.txt','a') as f:
        np.savetxt(f, Dq_i_use)
    with open('mu_v_loss.txt','a') as f:
        np.savetxt(f, mu_v_use)
    with open('mu_l_loss.txt','a') as f:
        np.savetxt(f, mu_l_use)
    with open('mu_i_loss.txt','a') as f:
        np.savetxt(f, mu_i_use)
    with open('mu_d_loss.txt','a') as f:
        np.savetxt(f, mu_d_use)
    with open('rho_loss.txt','a') as f:
        np.savetxt(f, rho_use)
    with open('temp_loss.txt','a') as f:
        np.savetxt(f, temp_use)
    with open('eta_loss.txt','a') as f:
        np.savetxt(f, eta_use)
    with open('Deta_loss.txt','a') as f:
        np.savetxt(f, Deta_use)
    with open('qd_loss.txt','a') as f:
        np.savetxt(f, qd_use)
    with open('qv_loss.txt','a') as f:
        np.savetxt(f, qv_use)
    with open('ql_loss.txt','a') as f:
        np.savetxt(f, ql_use)
    with open('qi_loss.txt','a') as f:
        np.savetxt(f, qi_use)
