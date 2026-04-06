#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 11:08:56 2025

@author: Rich
"""

import sys, os
sys.path.insert(0, os.path.expanduser("~/onezone/onezonerp"))
import eos
import numpy as np
from math import exp, pi
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#---constants (all in cgs)---
G = 6.6743e-8
M_sun = 1.98847e33
k_B = 1.380649e-16
m_u = 1.660539e-24
m_p = 1.6726219e-24
m_e = 9.10938356e-28
c = 3.0e10
a_rad = 7.5646e-15

h = 6.62607015e-27
hbar = h/(2*np.pi)
e_cgs = 4.80320425e-10


#---Neutron star parameters---
M_ns = 1.4 * M_sun
R_ns = 1.0e6
g_ns = G * M_ns / (R_ns**2) # take as constant

Mdot = 7.5e5

# Triple-alpha Q-value (7.274 MeV per reaction)
Q_3alpha = 7.275e6 * 1.602e-12  # erg

# Effective adiabatic gradient (degeneracy reduces from 0.4)
nabla_ad = 0.2


#---equaitons for scalar values---
def epsilon_3alpha(T, rho, Y_he):
    T8   = T * 1e-8
    rho5 = rho * 1e-5
    return 5.3e21 * (rho5**2) * (Y_he**3) * (T8**-3) * exp(-44.0 / T8)


def find_density(P_target, T, Ye, Yi):
    rho_lo, rho_hi = 1e-8, 1e14
    for _ in range(60):
        rho_mid = 0.5 * (rho_lo + rho_hi)
        P_mid = eos.pressure(rho_mid, Ye, Yi, T)
        if P_mid > P_target:
            rho_hi = rho_mid
        else:
            rho_lo = rho_mid
    return 0.5 * (rho_lo + rho_hi)

#---Duffusive coefficient start here---
def thermal_conductivity(rho, T, Ye, Yi):
    n_e = Ye * rho / m_u
    n_i = Yi * rho / m_u
    nu_c = 1e18 * (rho / 1e9) * (T / 1e8)**-2
    return (n_e**2 * k_B**2 * T) / (n_i * m_e * nu_c)

def kappa_es(T, rho, X_H=0.0):
    term_deg = 1.0 + 2.7e11 * (rho / (T**2))
    term_kn  = 1.0 + (T / 4.5e8)**0.86
    return 0.2 * (1.0 + X_H) / (term_deg * term_kn)

def kappa_ff(T, rho, Ye, X_he):
    mu_e = 1.0 / Ye
    T8   = T * 1e-8
    rho5 = rho * 1e-5
    sum_Z2_X_over_A = 3.0 - 2.0 * X_he
    return 0.753 * rho5 * (T8**(-3.5)) * mu_e * sum_Z2_X_over_A

def kappa_cond_from_Kcond(rho, T, Kcond):
    return (4.0 * a_rad * c * T**3) / (3.0 * rho * Kcond)

def diffusion_K_total(rho, T, Ye, Yi, X_he):
    # number densities
    n_e = Ye * rho / m_u
    n_i = Yi * rho / m_u

    # see eq 10-15
    p_F = hbar * (3.0 * np.pi**2 * n_e)**(1.0/3.0)
    E_F = np.sqrt((p_F * c)**2 + (m_e * c**2)**2) - m_e * c**2
    m_star = m_e + E_F / (c**2)

    Y_i_He = X_he / 4.0
    Y_i_C  = (1.0 - X_he) / 12.0
    sum_Yi_Z2 = 4.0 * Y_i_He + 36.0 * Y_i_C

    # collision frequency
    nu_c = (4.0 * e_cgs**4 * m_star) / (3.0 * np.pi * hbar**3) * (sum_Yi_Z2 / Ye)

    
    #---adding opacities---
    Kcond = (n_e**2 * k_B**2 * T) / (n_i * m_star * nu_c)

    k_es = kappa_es(T, rho, X_H=0.0)
    k_ff = kappa_ff(T, rho, Ye, X_he)
    k_cond = kappa_cond_from_Kcond(rho, T, Kcond)

    k_rad = k_es + k_ff
    k_tot = 1.0 / (1.0 / k_rad + 1.0 / k_cond)

    # diffusion coefficient
    K_tot = (4.0 * a_rad * c * T**3) / (3.0 * rho * k_tot)
    return K_tot

def specific_heat_cp(Yi):
    mu_i = 1.0 / Yi
    return 2.5 * k_B / (mu_i * m_p)

#---solve the ODE---
def derivatives(y, state):
    """
    in column depth, solving for [T, F, X_he] using eq.6-8
    """
    T, F, X_he = state
    X_he = max(X_he, 0.0) # just in case this fraction below 0 by possible rounding error

    # He (A=4, Z/A=0.5) and C (A=12, Z/A=0.5) => Ye = 0.5 throughout
    Ye = 0.5
    Yi = X_he/4.0 + (1.0 - X_he)/12.0

    # 1) in 2.1; for g = const: P = gy
    P = g_ns * y

    #---temrs for thermal balance---
    rho = find_density(P, T, Ye, Yi)
    cp = specific_heat_cp(Yi)
    eps = epsilon_3alpha(T, rho, X_he)
    K = diffusion_K_total(rho, T, Ye, Yi, X_he)

    # 2) in 2.1
    dT_dy = F / (rho * K)
    dF_dy = cp * Mdot * dT_dy - cp * (Mdot / y) * T * nabla_ad - eps

    # 3) in 2.1
    mass_he_per_reaction = 3 * 4 * m_p 
    Q_per_gHe = Q_3alpha / mass_he_per_reaction
    dX_dy = - eps / Q_per_gHe * (1.0 / Mdot)

    return [dT_dy, dF_dy, dX_dy]

#---integrate and plot---
y_start = 1e6
y_stop = 1e10
T_init = 2e8
F_init = 0.8*1.6*10**(-6)*Mdot/m_p
X_init = 1

sol = solve_ivp(
    derivatives,
    [y_start, y_stop],
    [T_init, F_init, X_init],
    method="BDF",
    rtol=1e-6, atol=1e-8,
    max_step=5e7
)

y_points = sol.t
T_profile, F_profile, X_profile = sol.y

fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True)

for ax in axes:
    ax.set_xscale("log")
    ax.set_xlim(y_start, y_stop)
    ax.xaxis.get_offset_text().set_fontsize(16)
    ax.yaxis.get_offset_text().set_fontsize(16)

axes[0].plot(y_points, T_profile)
axes[0].set_ylabel("$T$ (K)", fontsize = 20)
axes[0].grid(True, alpha=0.3)

axes[1].plot(y_points, X_profile)
axes[1].set_ylabel(r"$X_{\rm He}$", fontsize = 20)
axes[1].grid(True, alpha=0.3)

axes[2].plot(y_points, F_profile)
axes[2].set_ylabel(r"$F$ (erg cm$^{-2}$ s$^{-1}$)", fontsize = 20)
axes[2].set_xlabel(r"$y$ (g cm$^{-2}$)", fontsize = 20)
axes[2].grid(True, alpha=0.3)

for ax in axes:
    ax.set_xscale("log")
    ax.set_xlim(y_start, y_stop)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    
axes[1].set_yscale("log")

plt.savefig("minial_set_TXF.pdf", bbox_inches='tight')
plt.show()
