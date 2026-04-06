import dedalus.public as d3
import numpy as np
from math import sqrt, exp
from scipy.optimize import brentq
import time
import logging
from mpi4py import MPI

import scipy.sparse as sp
# Compatibility patch for Dedalus + SciPy >= 1.14
if not hasattr(sp.csr_matrix, "A"):
    sp.csr_matrix.A = property(lambda self: self.toarray())

#---if there is any numerical problem kill the process---
logger = logging.getLogger(__name__)
np.seterr(over='raise', divide='raise', invalid='raise')

#---constants in cgs---
arad = 7.5657e-15
c = 3.0e10
k_B = 1.38e-16
m_p = 1.67e-24
m_e = 9.10938356e-28
m_u = 1.660539e-24
hbar = 6.62607015e-27/(2*np.pi)
e_cgs= 4.80320425e-10
G = 6.6743e-8
rho0 = 1e6

nu_u = 1e30
dty_frac = 0.01# allow 1% change
nu_y = 1e22
nu_T = 1e22

nu_u6 = 1e22


#---NS parameters---
g = 1.90e14
mdot_Edd = 8.8e4
R_ns = 1.1e6  

X_h = 0.70
X_he = 0.30
Ye = 1.0 - X_he/2.0
Yi = 1.0 - 3.0*X_he/4.0

MeV_to_erg_per_nucleon = 1.602e-6
Q3alpha = 7.275 * MeV_to_erg_per_nucleon / m_p
Qbase = 0.15  * MeV_to_erg_per_nucleon / m_p

kappa_frac = 1
mdot0 = 3.2
Omega = 2*np.pi*100
A = 1e6 # for utheta
B = 1e4 # for uphi

# Gaussian parameters
mu = np.pi/2 # center at equator
sigma = 0.3 # width of Gaussian

SIM_T = 50*60 #seconds
SIM_DT = 5e-1


#---setup the physical quantities---
def pressure(rho, T):
    # degenerate + thermal electrons
    P_deg     = 9.91e12 * (Ye * rho)**(5.0/3.0)
    P_therm_e = Ye * rho * k_B * T / m_p
    P_e       = sqrt(P_deg**2 + P_therm_e**2)
    # ion ideal gas
    P_i  = Yi * rho * k_B * T / m_p
    # radiation
    P_rad = arad * T**4 / 3.0
    #return P_deg
    return P_e + P_i + P_rad

def find_rho_eqn(rho,P,T):
    return pressure(rho,T)-P
    
def find_density(P, T):
    '''
    do a numerical inversion to find density
    '''
    #return (P / (9.91e12))**(3/5) / Ye
    rho = brentq(find_rho_eqn,1.0,1e12,xtol=1e-6,args=(P,T))
    return rho

def kappa_es(T, rho, X_H):
    term_deg = 1.0 + 2.7e11*(rho/(T**2))
    term_kn  = 1.0 + (T/4.5e8)**0.86
    return 0.2*(1.0 + X_H)/(term_deg*term_kn)

def kappa_ff(T, rho, Ye_local, X_he_local):
    mu_e = 1.0/Ye_local
    T8   = T*1e-8
    rho5 = rho*1e-5
    YHe  = X_he_local/4.0
    YC   = (1.0 - X_he_local)/12.0
    sumZ = 4.0*YHe + 36.0*YC
    return 0.753*rho5*(T8**-3.5)*mu_e*sumZ

def Kcond_electron(rho, T, Ye_local, Yi_local, X_he_local):
    """Electron thermal conductivity K_cond [erg cm^-1 s^-1 K^-1].

    This is the same microphysical conductivity used to define the conduction opacity
    and (optionally) a thermal diffusivity kappa_th = K_cond / (rho * c_p).
    """
    n_e = Ye_local*rho/m_u
    n_i = Yi_local*rho/m_u
    p_F = hbar*(3*np.pi**2*n_e)**(1.0/3.0)
    E_F = sqrt((p_F*c)**2 + (m_e*c**2)**2) - m_e*c**2
    m_s = m_e + E_F/c**2
    YHe = X_he_local/4.0
    YC = (1.0 - X_he_local)/12.0
    sumZ = 4.0*YHe + 36.0*YC
    nu_ei = (4*e_cgs**4*m_s)/(3*np.pi*hbar**3)*(sumZ/Ye_local)
    Kcond = ( (n_e**2)*k_B**2*T )/(n_i*m_s*nu_ei)
    return Kcond

def kappa_cond(rho, T, Ye_local, Yi_local, X_he_local):
    """Effective 'conduction opacity' kappa_cond [cm^2 g^-1] corresponding to Kcond_electron."""
    Kcond = Kcond_electron(rho, T, Ye_local, Yi_local, X_he_local)
    return (4*arad*c*T**3)/(3*rho*Kcond)

def kappa_total(T, rho):
    k_es = kappa_es(T, rho, X_h)
    k_ff = kappa_ff(T, rho, Ye, X_he)
    k_r = k_es + k_ff
    k_c = kappa_cond(rho, T, Ye, Yi, X_he)
    # harmonic sum: 1/kappa = 1/k_r + 1/k_c
    #return 0.2
    return 1.0/(1.0/k_r + 1.0/k_c)

def epsilon_3alpha(T, rho, X_he_local):
    T8 = T*1e-8
    rho5 = rho*1e-5
    eps = 5.3e21*(rho5**2)*(X_he_local**3)*(T8**-3)*exp(-44.0/T8)
    eps *= (1.6 + (1.0 - X_he_local)*4.9)/0.606 
    #return 0
    return eps

def epsilon_total(T, y, rho, X_he_local, mdot_frac_local):
    eps = epsilon_3alpha(T, rho, X_he_local)
    eps += 5.8e15*0.01  
    eps += Qbase * (mdot_frac_local*mdot_Edd)/y
    return eps

def cp_total(T, rho):
    c_p = 2.5*(Yi + Ye)*k_B/m_p
    T8 = T*1e-8
    rho5 = rho*1e-5
    c_p += 4.0*arad*T8**3*1e19/rho5
    return c_p

#---set up the Dedalus integrator on Spherical coords---
Nphi, Ntheta = 64, 512
dealias = 3/2
dtype = np.float64

coords = d3.S2Coordinates('phi','theta')
dist = d3.Distributor(coords, dtype=dtype)
basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R_ns, dealias=dealias, dtype=dtype)

#---fields that evaluates with the PDE---
u = dist.VectorField(coords, name='u', bases=basis)

u_theta_vec = dist.VectorField(coords, name='u_theta_vec', bases=basis)

def sync_u_theta_vec():
    # keep CFL velocities at scale=1
    u.change_scales(1)
    u_theta_vec.change_scales(1)

    # copy theta component
    u_theta_vec['g'][:] = 0.0
    u_theta_vec['g'][1] = u['g'][1]    # S2: [0]=phi, [1]=theta

T = dist.Field(name='T', bases=basis)
y = dist.Field(name='y', bases=basis)
#y_dot = dist.Field(name='y_dot', bases=basis)

eps_nuc = dist.Field(name='eps_nuc', bases=basis)
flux = dist.Field(name='flux', bases=basis)
cp_fld = dist.Field(name='cp', bases=basis)
rho_fld = dist.Field(name='rho', bases=basis)

kappa_th_fld = dist.Field(name='kappa_th', bases=basis)

u_mag = dist.Field(name='u_mag', bases=basis)
u_phi_mag   = dist.Field(name='u_phi_mag', bases=basis)
u_theta_mag = dist.Field(name='u_theta_mag', bases=basis)

# Initial conditions
phi, theta = dist.local_grids(basis)
lat = np.pi/2 - theta
T['g'] = 2e8
y['g'] = 2e8
u['g'][...] = 0
'''
u.change_scales(1)  # keep ICs at scale=1
u['g'][0] = B * np.sin(theta)        # u_phi_0 = B sin(theta)
u['g'][1] = -A * np.sin(2.0*theta)   # u_theta_0 = -A sin(2 theta)
'''

mdot_frac = dist.Field(name='mdot_frac', bases=basis)
aaa = 1e-1


gaussian_profile = np.exp(-((theta - mu)**2) / (2*sigma**2))

# Apply Gaussian modulation to mdot_frac
# mdot_frac['g'] = mdot0 * (1 * gaussian_profile)
# mdot_frac['g'] = mdot0 * (1 + aaa*(np.sin(theta)))
mdot_frac['g'] = mdot0
'''
#---centrifugal---
phi_cen = dist.Field(name='phi_cen', bases=basis)
phi_cen['g'] = -0.5*(Omega**2)*(R_ns**2)*(np.sin(theta)**2)
'''

'''
#---temperature pertubation---
mu = np.cos(theta)                                   
deltaT = 2.0e3                                       
T['g'] += deltaT * 0.5 * (3*mu**2 - 1)
'''

#---update temperature and y---

def update_aux_fields():
    # Force consistent grid shapes (use scale=1 or scale=dealias, but be consistent)
    for fld in (T, y, mdot_frac, eps_nuc, flux, cp_fld, rho_fld, kappa_th_fld):
        fld.change_scales(dealias)

    Tg = T['g']; yg = y['g']
    yg_clip = np.where(yg > 1e-20, yg, 1e-20)
    Tg_clip = np.where(Tg > 1e-10, Tg, 1e-10)

    # pull the mdot map on the local grid
    mdotg = mdot_frac['g']

    eps_arr = np.empty_like(Tg_clip)
    flux_arr = np.empty_like(Tg_clip)
    cp_arr = np.empty_like(Tg_clip)
    rho_arr = np.empty_like(Tg_clip)
    kappa_th_arr = np.empty_like(Tg_clip)

    for idx in np.ndindex(Tg_clip.shape):
        Tloc = float(Tg_clip[idx]); yloc = float(yg_clip[idx])
        rho = find_density(g*yloc, Tloc)

        mdot_loc = float(mdotg[idx])

        eps = epsilon_total(Tloc, yloc, rho, X_he, mdot_loc)
        kap = kappa_total(Tloc, rho)
        # kap = 0.2
        cp_l = cp_total(Tloc, rho)
        # Thermal diffusivity for optional lateral diffusion: kappa_th = K_cond / (rho * c_p)
        # Thermal diffusivity from the total diffusion opacity (radiative + conductive)
        # Using radiative-diffusion relation: K_eff = 4 a c T^3 / (3 rho kappa_tot)
        Keff = (4.0*arad*c*(Tloc**3)) / (3.0*rho*kap)   # [erg cm^-1 s^-1 K^-1]
        kappa_th = Keff / (rho * cp_l) * kappa_frac                # [cm^2 s^-1]
        F = (c*arad*Tloc**4)/(3.0*kap*yloc)

        eps_arr[idx] = eps
        flux_arr[idx] = F
        cp_arr[idx] = cp_l
        rho_arr[idx]  = rho
        kappa_th_arr[idx] = kappa_th

    eps_nuc['g'] = eps_arr
    flux['g'] = flux_arr
    cp_fld['g'] = cp_arr
    rho_fld['g'] = rho_arr
    kappa_th_fld['g'] = kappa_th_arr

update_aux_fields()

#---update burning rate---
eps3 = dist.Field(name='eps3', bases=basis)

def update_eps3_field():
    for fld in (T, y, eps3):
        fld.change_scales(dealias)
        
    Tg = T['g']; yg = y['g']
    yg_clip = np.where(yg > 1e-20, yg, 1e-20)
    Tg_clip = np.where(Tg > 1e-10, Tg, 1e-10)

    arr = np.empty_like(Tg_clip)
    it = np.ndindex(Tg_clip.shape)
    for idx in it:
        Tloc = float(Tg_clip[idx]); yloc = float(yg_clip[idx])
        rho = find_density(g*yloc, Tloc)
        arr[idx] = epsilon_3alpha(Tloc, rho, X_he)
    eps3['g'] = arr

update_eps3_field()

def update_u_mag():
    # Make sure everything is at the same scale
    u.change_scales(1)
    u_mag.change_scales(1)
    u_phi_mag.change_scales(1)
    u_theta_mag.change_scales(1)

    ug = u['g']  # shape: (n_components, Nphi_loc, Ntheta_loc)

    # On S2Coordinates('phi','theta'), components are [u_phi, u_theta]
    u_phi_mag['g']   = np.abs(ug[0])
    u_theta_mag['g'] = np.abs(ug[1])

    # Total magnitude
    u_mag['g'] = np.sqrt(np.sum(ug**2, axis=0))

update_u_mag()

#---setup the PDE with defined scalar + vector fields---
problem = d3.IVP([u, T, y], namespace=locals())

#---with rotation---
problem.add_equation("dt(u) + nu_u*(lap(lap(u))) + 2*Omega*MulCosine(skew(u)) = -g*grad(y)/rho_fld - u@grad(u)")
#problem.add_equation("dt(u) + nu_u6*lap((lap(lap(u)))) + 2*Omega*MulCosine(skew(u)) = -g*grad(y)/rho_fld - u@grad(u)")
#problem.add_equation("dt(y) = -u@grad(y) + mdot_frac*mdot_Edd - 12*eps3*y/Q3alpha")
problem.add_equation("dt(y) = -div(u*y) + mdot_frac*mdot_Edd - 12*eps3*y/Q3alpha")
problem.add_equation("dt(T) = -u@grad(T) + (eps_nuc - flux/y)/cp_fld + kappa_th_fld*lap(T)")

#---solve the system---
solver = problem.build_solver(d3.SBDF2)
# solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = SIM_T

snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=SIM_DT, max_writes=1e9)
snapshots.add_task(flux, name="flux") # for plot_sphere_flux.py
snapshots.add_task(T, name="T")
snapshots.add_task(kappa_th_fld, name="kappa_th")
snapshots.add_task(y, name="y")
snapshots.add_task(
    - d3.div(y*u) + mdot_frac*mdot_Edd - 12*eps3*y/Q3alpha,
    name="y_dot",
)
snapshots.add_task(u_mag, name="u") # save u_magnitude as scalar field for easy plotting
snapshots.add_task(u_phi_mag, name="u_phi_mag")
snapshots.add_task(u_theta_mag, name="u_theta_mag")

#---CFL dt, the stable dt by CFL turns out to be ~9.9e-2---
CFL = d3.CFL(
    solver,
    initial_dt=1e-8,
    cadence=1,
    safety=0.3,
    threshold=0.1,
    max_change=2.0,
    min_change=0.1,
    max_dt=SIM_DT
)
CFL.add_velocity(u)

'''
u_eff = 2*np.pi*R_ns/512 * ((y_dot)) / (y)
u_y = u_eff * e_theta
CFL2 = d3.CFL(
    solver,
    initial_dt=1e-8,
    cadence=1,
    safety=0.01,
    threshold=0.1,
    max_change=2.0,
    min_change=0.1,
    max_dt=1
)

CFL2.add_velocity(u_y)
'''


# CFL.add_velocity(u_theta_vec)

# After you have coords, basis, dist, u, R_ns, etc.
phi, theta = dist.local_grids(basis)

y_rhs = - d3.div(y*u) + mdot_frac*mdot_Edd - 12*eps3*y/Q3alpha


# Print size of each terms in u_momentum

TERM_PRINT_EVERY = 50  # iterations (set None/0 to disable)

def _vec_stats(vfld, scale=dealias):
    """Global max(|component|) and global RMS for a Dedalus VectorField."""
    vfld.change_scales(scale)
    vg = vfld['g']  # shape (ncomp, nphi_loc, ntheta_loc)

    local_max = np.max(np.abs(vg))
    local_sum2 = np.sum(vg * vg)
    local_n = vg.size

    gmax = dist.comm.allreduce(local_max, op=MPI.MAX)
    gsum2 = dist.comm.allreduce(local_sum2, op=MPI.SUM)
    gn = dist.comm.allreduce(local_n, op=MPI.SUM)

    grms = np.sqrt(gsum2 / max(gn, 1))
    return gmax, grms

def _fld_stats(fld, scale=dealias):
    """Global max(|field|) and global RMS for a Dedalus scalar Field."""
    fld.change_scales(scale)
    fg = fld['g']

    local_max = np.max(np.abs(fg))
    local_sum2 = np.sum(fg * fg)
    local_n = fg.size

    gmax = dist.comm.allreduce(local_max, op=MPI.MAX)
    gsum2 = dist.comm.allreduce(local_sum2, op=MPI.SUM)
    gn = dist.comm.allreduce(local_n, op=MPI.SUM)

    grms = np.sqrt(gsum2 / max(gn, 1))
    return gmax, grms


def print_u_term_sizes(dt_current):
    """
    Evaluate each term in:
        dt(u) + nu_u*lap(lap(u)) + 2*Omega*MulCosine(skew(u)) = - g*grad(y)/rho_fld - u@grad(u)
    and print comparable magnitudes (global max and RMS).
    """
    # Consistent scale for fair comparison
    u.change_scales(dealias)
    y.change_scales(dealias)

    term_visc = (nu_u * d3.lap(d3.lap(u))).evaluate()
    #term_visc = (nu_u6 * d3.lap(d3.lap(d3.lap(u)))).evaluate()
    # Coriolis term: avoid Omega==0 turning the whole expression into an int/float
    cor_op = d3.MulCosine(d3.skew(u)).evaluate()   # VectorField
    cor_op.change_scales(dealias)
    cor_op['g'] *= float(2.0 * Omega) # safe for Omega==0
    term_cor = cor_op
    term_pres = (-g * d3.grad(y) / rho_fld).evaluate()
    term_adv  = (-(u @ d3.grad(u))).evaluate()

    rhs_forcing = (term_pres + term_adv).evaluate()
    rhs_total   = (term_pres + term_adv - term_visc - term_cor).evaluate()

    mv, rv = _vec_stats(term_visc)
    mc, rc = _vec_stats(term_cor)
    mp, rp = _vec_stats(term_pres)
    ma, ra = _vec_stats(term_adv)
    mF, rF = _vec_stats(rhs_forcing)
    mT, rT = _vec_stats(rhs_total)

    if dist.comm.rank == 0:
        print(
            f"[u-term-sizes] it={solver.iteration:7d} t={solver.sim_time:10.3e} dt={dt_current:9.3e}\n"
            f"  visc : max={mv:12.4e}  rms={rv:12.4e}\n"
            f"  cor  : max={mc:12.4e}  rms={rc:12.4e}\n"
            f"  pres : max={mp:12.4e}  rms={rp:12.4e}\n"
            f"  adv  : max={ma:12.4e}  rms={ra:12.4e}\n"
            f"  RHS(pres+adv): max={mF:12.4e}  rms={rF:12.4e}\n"
            f"  RHS(total)   : max={mT:12.4e}  rms={rT:12.4e}\n",
            flush=True
        )


def print_y_div_terms(dt_current):
    """
    Split the mass-flux divergence via the product rule:
        -div(y*u) = -(u@grad(y)) - y*div(u)
    and print which piece dominates (global max and RMS).

    Note: on the sphere, grad/div are the Dedalus S2 operators, so this is the
    *exact* product-rule split in the same geometry.
    """
    # Consistent scale for fair comparison
    u.change_scales(dealias)
    y.change_scales(dealias)

    term_adv = (-(u @ d3.grad(y))).evaluate()          # scalar Field
    term_comp = (-(y * d3.div(u))).evaluate()          # scalar Field
    term_sum = (term_adv + term_comp).evaluate()       # should match -div(y*u)
    term_direct = (-(d3.div(y*u))).evaluate()          # direct operator form (sanity check)

    ma, ra = _fld_stats(term_adv)
    mc, rc = _fld_stats(term_comp)
    ms, rs = _fld_stats(term_sum)
    md, rd = _fld_stats(term_direct)

    if dist.comm.rank == 0:
        print(
            f"[y-div-split] it={solver.iteration:7d} t={solver.sim_time:10.3e} dt={dt_current:9.3e}\n"
            f"  -(u·grad y)   : max={ma:12.4e}  rms={ra:12.4e}\n"
            f"  -(y div u)    : max={mc:12.4e}  rms={rc:12.4e}\n"
            f"  split sum     : max={ms:12.4e}  rms={rs:12.4e}\n"
            f"  direct -div(yu): max={md:12.4e}  rms={rd:12.4e}\n",
            flush=True
        )


dt = 1e-8  # initial dt

# --- main loop ---
start_time = time.time()
try:
    logger.info('Starting main loop (1-D theta)')
    while solver.proceed:
        # update microphysics from current (T,y) BEFORE taking the step
        update_aux_fields()
        update_eps3_field()
        sync_u_theta_vec()

        # 1) CFL suggestion
        dt_cfl = CFL.compute_timestep()

        #dt_y_cfl = CFL2.compute_timestep()

        # 2) evaluate y_t
        y_t = y_rhs.evaluate()
        y_t.change_scales(dealias)
        y.change_scales(dealias)

        y_g   = y['g']
        y_t_g = y_t['g']

        y_min = np.min(y_g)
        if y_min < 0:
            raise ValueError(f"ERROR: y has become negative! Minimum y = {y_min:.3e}")

        # 3) pointwise ratio |y_t / y|
        tiny = 1e-30  # floor to avoid division by zero
        ratio = np.abs(y_t_g) / np.maximum(np.abs(y_g), tiny)

        max_ratio = np.max(ratio)

        frac = dty_frac
        dt_y_local = frac / max_ratio
        
        dt_y = dist.comm.allreduce(dt_y_local, op=MPI.MIN)

        # 4) combine constraints
        dt = min(dt_cfl, dt_y)
        if_dty = dt == dt_y

        solver.step(dt)
        
        if TERM_PRINT_EVERY and (solver.iteration - 1) % TERM_PRINT_EVERY == 0:
            print_u_term_sizes(dt)
            print_y_div_terms(dt)
        
        update_u_mag()

        if (solver.iteration - 1) % 10 == 0:
            if if_dty:
                print(f"Iter={solver.iteration:6d}  t={solver.sim_time:9.3e}  dty={dt:8.3e}",
                    flush=True)
            else:
                print(f"Iter={solver.iteration:6d}  t={solver.sim_time:9.3e}  dt={dt:8.3e}",
                    flush=True)

except Exception as e:
    logger.error('Exception during main loop:', exc_info=e)
    raise
finally:
    solver.log_stats()
    print(f"Elapsed wall time: {time.time() - start_time:.2f} s")


