# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 0  (chạy trước 1 lần duy nhất)                                   ║
# ║  !pip install cupy-cuda12x -q                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 1  —  paste toàn bộ file này vào 1 cell Colab / Kaggle           ║
# ║  Estimated runtime: ~15 min (T4 GPU)                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ════════════════════════════════════════════════════════════════════════════
# 0.  IMPORTS + GPU
# ════════════════════════════════════════════════════════════════════════════
import os, math, time, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
warnings.filterwarnings("ignore")

try:
    import cupy as cp
    xp  = cp
    GPU = True
    print("✓  CuPy detected — GPE runs on GPU")
except ImportError:
    xp  = np
    GPU = False
    print("⚠  CuPy not found — falling back to NumPy (slower)")

def to_np(a):
    return cp.asnumpy(a) if (GPU and isinstance(a, cp.ndarray)) else np.asarray(a)

np.random.seed(20260519)
if GPU: cp.random.seed(20260519)

# ════════════════════════════════════════════════════════════════════════════
# 1.  PHYSICAL CONSTANTS  (SI)
# ════════════════════════════════════════════════════════════════════════════
hbar     = 1.054571817e-34       # J·s
kB       = 1.380649e-23          # J/K
muB_g    = 1.4e10                # g·µB/h  [Hz/T]
# YIG parameters
Ms       = 1.4e5                 # A/m
alpha_G  = 2.0e-4                # Gilbert damping
omega0   = 2.0*np.pi*3.75e9     # BEC frequency [rad/s]
q_BEC    = 1.8e7                 # BEC wave-vector [rad/m]
D_zz     = 3.0e-4               # Dispersion coeff [m²/s]
T_self   = 1.0e-10              # GP self-interaction [m/s]
S_cross  = 0.7e-10              # GP cross-interaction [m/s]
Gamma0   = alpha_G * omega0     # Gilbert relaxation [rad/s]
T_K      = 300.0                 # Temperature [K]

print(f"   omega0/2π = {omega0/(2*np.pi)/1e9:.3f} GHz")
print(f"   Gamma0/2π = {Gamma0/(2*np.pi)/1e6:.3f} MHz")

# ════════════════════════════════════════════════════════════════════════════
# 2.  SPATIAL GRID
# ════════════════════════════════════════════════════════════════════════════
L_um  = 20.0
Nz    = 256
Lphys = 2.0*L_um*1e-6
dz    = Lphys/Nz
z_np  = (np.arange(Nz) - Nz/2)*dz
z_um  = z_np*1e6
kz_np = 2.0*np.pi*np.fft.fftfreq(Nz, d=dz)

z_xp  = xp.asarray(z_np)
kz_xp = xp.asarray(kz_np)

mask_A = xp.asarray(z_np < 0).astype(xp.float64)
mask_B = xp.asarray(z_np > 0).astype(xp.float64)

# ════════════════════════════════════════════════════════════════════════════
# 3.  ANALYTICAL THEORY  (Part 1)
# ════════════════════════════════════════════════════════════════════════════
def K_WKB(d, dB):
    """Josephson coupling from WKB tunnelling. d [m], dB [T]."""
    V0  = 2.0*np.pi*muB_g*dB*hbar
    kap = np.sqrt(np.maximum(V0, 1e-30)/(hbar*D_zz))
    return (omega0/np.pi)*np.exp(-kap*d)

def F_bound(K):
    """Population fidelity bound  F ≥ 1 - πΓ/K  [from exp(-2Γτ_g) ≈ 1-2Γτ_g]."""
    return 1.0 - np.pi*Gamma0/np.maximum(K, 1.0)

def tau_gate(K):
    """Half-Rabi gate time  τ = π/(2K) [s]."""
    return np.pi/(2.0*np.maximum(K, 1.0))

K_star = np.pi*Gamma0/0.01               # K for F≥0.99  → K*/2π ≈ 236 MHz
K_demo = 2.0*np.pi*250e6                 # operating point (> K*), τ_g = 1.0 ns
T_demo = tau_gate(K_demo)*1e9             # ns
print(f"   K*/2π  = {K_star/(2*np.pi)/1e6:.1f} MHz  (F≥0.99 threshold)")
print(f"   K_demo = {K_demo/(2*np.pi)/1e6:.1f} MHz   τ_gate = {T_demo:.2f} ns")

# ════════════════════════════════════════════════════════════════════════════
# 4.  GPE HELPERS
# ════════════════════════════════════════════════════════════════════════════
def make_potential(d_bar=1e-6, dB_bar=5e-3):
    """Double-well trapping potential V(z) [rad/s]."""
    omega_trap = 2.0*np.pi*50e6
    z_well     = (L_um*1e-6)/2.0
    delta2     = xp.minimum((z_xp - z_well)**2, (z_xp + z_well)**2)
    V          = (omega_trap**2/(4.0*D_zz))*delta2
    Omega_bar  = 2.0*np.pi*muB_g*dB_bar
    sigma      = dz*2
    bar = 0.5*(xp.tanh((z_xp+d_bar/2)/sigma) - xp.tanh((z_xp-d_bar/2)/sigma))
    return V + Omega_bar*bar

def gauss(center_um, sigma_um=3.0):
    """Normalised Gaussian wavepacket on z_xp."""
    g = xp.exp(-((z_xp - center_um*1e-6)**2)/(2*(sigma_um*1e-6)**2))
    return g/xp.sqrt(xp.sum(xp.abs(g)**2)*dz)

def init_state(ctrl, tgt):
    """Build (Ap, Am, Bp, Bm) initial wavefunctions."""
    cA, cB = -L_um/2, +L_um/2
    eps = 1e-3
    mk  = {'0': (gauss(cA),     eps*gauss(cA)),
           '1': (eps*gauss(cA), gauss(cA))}
    mb  = {'0': (gauss(cB),     eps*gauss(cB)),
           '1': (eps*gauss(cB), gauss(cB))}
    Ap, Am = mk[ctrl]; Bp, Bm = mb[tgt]
    return Ap, Am, Bp, Bm

def renorm(Pp, Pm, mask):
    n = xp.sqrt(xp.sum((xp.abs(Pp)**2 + xp.abs(Pm)**2)*mask)*dz + 1e-30)
    return Pp/n, Pm/n

# ════════════════════════════════════════════════════════════════════════════
# 5.  SPLIT-STEP GPE STEP
# ════════════════════════════════════════════════════════════════════════════
def gpe_step(psi, V, K_max, dt, noise_amp=0.0, do_renorm=True):
    """One Strang-split step for 4-component GPE."""
    Ap, Am, Bp, Bm = psi
    kin = xp.exp(-1j*D_zz*kz_xp**2*dt*0.5)

    # half kinetic
    Ap = xp.fft.ifft(kin*xp.fft.fft(Ap))
    Am = xp.fft.ifft(kin*xp.fft.fft(Am))
    Bp = xp.fft.ifft(kin*xp.fft.fft(Bp))
    Bm = xp.fft.ifft(kin*xp.fft.fft(Bm))

    # nonlinear + damping
    nAp = xp.abs(Ap)**2; nAm = xp.abs(Am)**2
    nBp = xp.abs(Bp)**2; nBm = xp.abs(Bm)**2
    Ap *= xp.exp((-1j*(V + T_self*nAp + S_cross*nAm) - Gamma0)*dt)
    Am *= xp.exp((-1j*(V + T_self*nAm + S_cross*nAp) - Gamma0)*dt)
    Bp *= xp.exp((-1j*(V + T_self*nBp + S_cross*nBm) - Gamma0)*dt)
    Bm *= xp.exp((-1j*(V + T_self*nBm + S_cross*nBp) - Gamma0)*dt)

    # conditional Josephson: K_eff from integrated control population
    NA_p = float(to_np(xp.sum(nAp*mask_A)))*dz
    NA_m = float(to_np(xp.sum(nAm*mask_A)))*dz
    K_eff = K_max * NA_m/(NA_p + NA_m + 1e-30)
    if K_eff > 0.0:
        cB = math.cos(K_eff*dt); sB = math.sin(K_eff*dt)
        Bp, Bm = cB*Bp - 1j*sB*Bm, cB*Bm - 1j*sB*Bp

    # half kinetic
    Ap = xp.fft.ifft(kin*xp.fft.fft(Ap))
    Am = xp.fft.ifft(kin*xp.fft.fft(Am))
    Bp = xp.fft.ifft(kin*xp.fft.fft(Bp))
    Bm = xp.fft.ifft(kin*xp.fft.fft(Bm))

    # Langevin noise
    if noise_amp > 0.0:
        sh = Ap.shape
        for arr in [Ap, Am, Bp, Bm]:
            arr += noise_amp*(xp.random.normal(0,1,sh)+1j*xp.random.normal(0,1,sh))

    # renormalise (continuous pumping)
    if do_renorm:
        Ap, Am = renorm(Ap, Am, mask_A)
        Bp, Bm = renorm(Bp, Bm, mask_B)

    return Ap, Am, Bp, Bm, K_eff

# ════════════════════════════════════════════════════════════════════════════
# 6.  RUN CNOT SIMULATION  (Part 2)
# ════════════════════════════════════════════════════════════════════════════
def run_sim(ctrl, tgt, K_max, T_ns, dt_ns=0.005, n_save=200,
            noise=True, do_renorm=True):
    dt     = dt_ns*1e-9
    Nt     = int(T_ns/dt_ns)
    save_e = max(1, Nt//n_save)
    V      = make_potential()
    n_amp  = (float(np.sqrt(Gamma0*kB*T_K/(hbar*omega0)))*1e-7*
              np.sqrt(dt)) if noise else 0.0

    Ap, Am, Bp, Bm = init_state(ctrl, tgt)
    t_s=[]; NAp_s=[]; NAm_s=[]; NBp_s=[]; NBm_s=[]; phi_s=[]; Keff_s=[]

    for it in range(Nt+1):
        if it % save_e == 0:
            t_s.append(it*dt_ns)
            NAp_s.append(to_np(xp.abs(Ap)**2).copy())
            NAm_s.append(to_np(xp.abs(Am)**2).copy())
            NBp_s.append(to_np(xp.abs(Bp)**2).copy())
            NBm_s.append(to_np(xp.abs(Bm)**2).copy())
            jB = int(Nz/2 + Nz/4)
            phi_s.append(float(to_np(xp.angle(Bp[jB]*xp.conj(Bm[jB])))))
        if it == Nt: break
        Ap, Am, Bp, Bm, Ke = gpe_step(
            (Ap,Am,Bp,Bm), V, K_max, dt, n_amp, do_renorm=do_renorm)
        if it % save_e == 0: Keff_s.append(Ke)

    return dict(t_ns=np.array(t_s), z_um=z_um,
                N_Ap=np.array(NAp_s), N_Am=np.array(NAm_s),
                N_Bp=np.array(NBp_s), N_Bm=np.array(NBm_s),
                phi_B=np.array(phi_s))

def integ(sim, region):
    """Spatially-integrated, normalised populations for qubit A or B."""
    mask = (sim['z_um'] < 0) if region=='A' else (sim['z_um'] > 0)
    Np = sim[f'N_{"A" if region=="A" else "B"}p'][:,mask].sum(1)*dz*1e6
    Nm = sim[f'N_{"A" if region=="A" else "B"}m'][:,mask].sum(1)*dz*1e6
    tot = Np + Nm + 1e-30
    return Np/tot, Nm/tot

truth_cases = [
    ('0','0',r'$|00\rangle\!\to\!|00\rangle$','0'),
    ('0','1',r'$|01\rangle\!\to\!|01\rangle$','1'),
    ('1','0',r'$|10\rangle\!\to\!|11\rangle$','1'),
    ('1','1',r'$|11\rangle\!\to\!|10\rangle$','0'),
]

print("\n── Part 2: Spatial GPE truth table ─────────────────────────────")
sims = {}
for ctrl, tgt, label, exp_B in truth_cases:
    t0 = time.time()
    sims[(ctrl,tgt)] = run_sim(ctrl, tgt, K_demo, T_demo,
                                dt_ns=0.005, n_save=200, noise=True)
    nBp, nBm = integ(sims[(ctrl,tgt)], 'B')
    F = nBp[-1] if exp_B=='0' else nBm[-1]
    print(f"  {label:35s}  F={F:.4f}   [{time.time()-t0:.1f}s]")

tt_F = []
for ctrl,tgt,_,exp_B in truth_cases:
    nBp,nBm = integ(sims[(ctrl,tgt)],'B')
    tt_F.append(nBp[-1] if exp_B=='0' else nBm[-1])
print(f"  F_avg (pumped) = {np.mean(tt_F):.4f}")

# ════════════════════════════════════════════════════════════════════════════
# 7.  PARAMETER SWEEP — REDUCED 4-MODE MODEL  (Part 3)
# ════════════════════════════════════════════════════════════════════════════
def rhs_4m(s, K_max, Gam):
    Ap=s[0]+1j*s[1]; Am=s[2]+1j*s[3]; Bp=s[4]+1j*s[5]; Bm=s[6]+1j*s[7]
    nA  = np.abs(Ap)**2+np.abs(Am)**2+1e-30
    Ke  = K_max*np.clip(np.abs(Am)**2/nA, 0, 1)
    dAp = (-1j*(T_self*np.abs(Ap)**2+S_cross*np.abs(Am)**2)-Gam)*Ap
    dAm = (-1j*(T_self*np.abs(Am)**2+S_cross*np.abs(Ap)**2)-Gam)*Am
    dBp = ((-1j*(T_self*np.abs(Bp)**2+S_cross*np.abs(Bm)**2)-Gam)*Bp
           -1j*Ke*Bm)
    dBm = ((-1j*(T_self*np.abs(Bm)**2+S_cross*np.abs(Bp)**2)-Gam)*Bm
           -1j*Ke*Bp)
    return np.array([dAp.real,dAp.imag,dAm.real,dAm.imag,
                     dBp.real,dBp.imag,dBm.real,dBm.imag])

def rk4_4m(s0, K_max, Gam, T_sim, Nt=300):
    dt=T_sim/Nt; s=s0.copy()
    for _ in range(Nt):
        k1=rhs_4m(s,K_max,Gam)
        k2=rhs_4m(s+.5*dt*k1,K_max,Gam)
        k3=rhs_4m(s+.5*dt*k2,K_max,Gam)
        k4=rhs_4m(s+   dt*k3,K_max,Gam)
        s += (dt/6)*(k1+2*k2+2*k3+k4)
    return s

def fidelity_4m(K_max, Gam=Gamma0):
    """Average truth-table fidelity (no renorm → damping penalises F)."""
    cases=[('0','0','0'),('0','1','1'),('1','0','1'),('1','1','0')]
    pure ={'0':(1+0j,1e-3+0j),'1':(1e-3+0j,1+0j)}
    T_s  = np.pi/(2*max(K_max,1.0))
    Fs=[]
    for c,t,eB in cases:
        Ap0,Am0=pure[c]; Bp0,Bm0=pure[t]
        s0=np.array([Ap0.real,Ap0.imag,Am0.real,Am0.imag,
                     Bp0.real,Bp0.imag,Bm0.real,Bm0.imag])
        sf=rk4_4m(s0,K_max,Gam,T_s)
        nBp=sf[4]**2+sf[5]**2; nBm=sf[6]**2+sf[7]**2
        Fs.append(nBp if eB=='0' else nBm)
    return float(np.mean(Fs))

print("\n── Part 3: Parameter sweeps ─────────────────────────────────────")
K_sw  = np.logspace(7, 9.5, 28)
G_sw  = np.logspace(5, 8,   24)
d_sw  = np.linspace(0.3, 3.0, 24)
dB_sw = np.linspace(0.5, 30.0, 24)

F_KG = np.array([[fidelity_4m(K,G) for G in G_sw] for K in K_sw])
F_Kd = np.zeros((len(d_sw), len(dB_sw)))
K_Kd = np.zeros_like(F_Kd)
for i,d in enumerate(d_sw):
    for j,dB in enumerate(dB_sw):
        Kv = K_WKB(d*1e-6, dB*1e-3)
        K_Kd[i,j] = Kv
        F_Kd[i,j] = fidelity_4m(min(Kv, 5e9))

idx = np.unravel_index(np.argmin(np.abs(K_Kd - K_star)), K_Kd.shape)
print(f"  Optimal: d={d_sw[idx[0]]:.2f}µm  dB={dB_sw[idx[1]]:.1f}mT"
      f"  K={K_Kd[idx]/(2*np.pi)/1e6:.1f}MHz  F={F_Kd[idx]:.4f}")

# ════════════════════════════════════════════════════════════════════════════
# 8.  ENSEMBLE FIDELITY  (500 trajectories, no renorm)
# ════════════════════════════════════════════════════════════════════════════
N_magnon = 1e10
sig_th   = float(np.sqrt(kB*T_K/(hbar*omega0*N_magnon)))

def rhs_4m_batch(s, K_max, Gam):
    """Vectorised RHS over batch axis 0. s: (N,8)"""
    Ap=s[:,0]+1j*s[:,1]; Am=s[:,2]+1j*s[:,3]
    Bp=s[:,4]+1j*s[:,5]; Bm=s[:,6]+1j*s[:,7]
    nA  = np.abs(Ap)**2+np.abs(Am)**2+1e-30
    Ke  = K_max*np.clip(np.abs(Am)**2/nA, 0, 1)
    dAp = (-1j*(T_self*np.abs(Ap)**2+S_cross*np.abs(Am)**2)-Gam)*Ap
    dAm = (-1j*(T_self*np.abs(Am)**2+S_cross*np.abs(Ap)**2)-Gam)*Am
    dBp = ((-1j*(T_self*np.abs(Bp)**2+S_cross*np.abs(Bm)**2)-Gam)*Bp
           -1j*Ke*Bm)
    dBm = ((-1j*(T_self*np.abs(Bm)**2+S_cross*np.abs(Bp)**2)-Gam)*Bm
           -1j*Ke*Bp)
    return np.stack([dAp.real,dAp.imag,dAm.real,dAm.imag,
                     dBp.real,dBp.imag,dBm.real,dBm.imag],axis=1)

def propagate_batch(s0, K_max, Gam, T_sim, Nt=400):
    dt=T_sim/Nt; s=s0.copy()
    for _ in range(Nt):
        k1=rhs_4m_batch(s,K_max,Gam)
        k2=rhs_4m_batch(s+.5*dt*k1,K_max,Gam)
        k3=rhs_4m_batch(s+.5*dt*k2,K_max,Gam)
        k4=rhs_4m_batch(s+   dt*k3,K_max,Gam)
        s += (dt/6)*(k1+2*k2+2*k3+k4)
    return s

def ensemble_F(K_max, Gam=Gamma0, N=500, noise=sig_th, Nt=400):
    pure  = {'0':(1+0j,1e-3+0j),'1':(1e-3+0j,1+0j)}
    cases = [('0','0','0'),('0','1','1'),('1','0','1'),('1','1','0')]
    T_s   = np.pi/(2*K_max)
    rng   = np.random.default_rng(42)
    Fc, Fe = [], []
    for c,t,eB in cases:
        Ap0,Am0=pure[c]; Bp0,Bm0=pure[t]
        def noisy(a):
            r=rng.normal(0,noise,(N,2))
            return complex(a)+r[:,0]+1j*r[:,1]
        Ap=noisy(Ap0);Am=noisy(Am0);Bp=noisy(Bp0);Bm=noisy(Bm0)
        nA=np.sqrt(np.abs(Ap)**2+np.abs(Am)**2+1e-30)
        nB=np.sqrt(np.abs(Bp)**2+np.abs(Bm)**2+1e-30)
        Ap/=nA;Am/=nA;Bp/=nB;Bm/=nB
        s0=np.stack([Ap.real,Ap.imag,Am.real,Am.imag,
                     Bp.real,Bp.imag,Bm.real,Bm.imag],axis=1)
        sf  = propagate_batch(s0,K_max,Gam,T_s,Nt)
        nBp = sf[:,4]**2+sf[:,5]**2; nBm=sf[:,6]**2+sf[:,7]**2
        # absolute population (no renorm → damping penalises F)
        Ft  = nBp if eB=='0' else nBm
        Fc.append(float(np.mean(Ft)))
        Fe.append(float(np.std(Ft)/np.sqrt(N)))
    return np.array(Fc),np.array(Fe),float(np.mean(Fc)),float(np.mean(Fe))

print("\n── Part 4: Ensemble fidelity (500 traj, 300K, no pump) ─────────")
t0=time.time()
F_ens, F_err, F_ens_mean, F_ens_std = ensemble_F(K_demo, N=500)
print(f"  Grand mean  F = {F_ens_mean:.4f} ± {F_ens_std:.4f}  [{time.time()-t0:.1f}s]")
for (c,t,_,eB),f,e in zip(truth_cases, F_ens, F_err):
    print(f"  {c}{t}→{'1' if eB=='1' else '0'}   F = {f:.4f} ± {e:.4f}")

# F vs noise
noise_sw = np.array([0.000,0.001,0.003,0.005,0.010,0.020])
F_n_sw   = [ensemble_F(K_demo,N=300,noise=n,Nt=300)[2] for n in noise_sw]
# F vs K
K_ens_sw = np.logspace(7.5,9.5,18)
F_K_sw   = [ensemble_F(K,N=200,Nt=250)[2] for K in K_ens_sw]
print("  Sweeps done.")

# ════════════════════════════════════════════════════════════════════════════
# 9.  UNIVERSALITY  (Part 5)
# ════════════════════════════════════════════════════════════════════════════
def Rx(t): c=math.cos(t/2);s=math.sin(t/2); return np.array([[c,-1j*s],[-1j*s,c]],complex)
def Rz(t): return np.diag([np.exp(-1j*t/2), np.exp(1j*t/2)])

H_mat    = Rz(np.pi/2)@Rx(np.pi/2)@Rz(np.pi/2)
S_mat    = Rz(np.pi/2)
T_mat    = Rz(np.pi/4)
I2       = np.eye(2,dtype=complex)
CNOT_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],complex)

H_tgt    = (1/np.sqrt(2))*np.array([[1,1],[1,-1]],complex)
H_ov     = abs(np.trace(H_mat.conj().T@H_tgt))/2
state00  = np.array([1,0,0,0],complex)
bell     = CNOT_mat@(np.kron(H_mat,I2)@state00)
bell_tgt = (1/np.sqrt(2))*np.array([1,0,0,1],complex)
bell_F   = abs(np.vdot(bell_tgt, bell))**2
print(f"\n── Part 5: Universality ────────────────────────────────────────")
print(f"  H overlap = {H_ov:.6f}")
print(f"  Bell F    = {bell_F:.6f}")

# ════════════════════════════════════════════════════════════════════════════
# 10. MATPLOTLIB STYLE
# ════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family':'serif','font.size':9,'axes.labelsize':10,
    'axes.titlesize':9,'legend.fontsize':8,'xtick.labelsize':8,
    'ytick.labelsize':8,'xtick.direction':'in','ytick.direction':'in',
    'xtick.top':True,'ytick.right':True,'axes.linewidth':0.8,
    'lines.linewidth':1.5,'savefig.dpi':300,'savefig.bbox':'tight',
    'savefig.pad_inches':0.05,
})
C1='#1f77b4'; C2='#d62728'; C3='#2ca02c'; CB='#ff7f0e'

# ════════════════════════════════════════════════════════════════════════════
# 11. FIGURE 1 — Analytical theory
# ════════════════════════════════════════════════════════════════════════════
print("\n── Figures ─────────────────────────────────────────────────────")
fig1,ax1=plt.subplots(2,2,figsize=(7,5.5))
fig1.subplots_adjust(hspace=0.44,wspace=0.38)

d_arr  = np.linspace(0.3,4.0,300)*1e-6
dB_arr = np.linspace(0.5,50.0,300)*1e-3
K_arr  = np.logspace(6.5,10,400)

ax=ax1[0,0]
ax.semilogy(d_arr*1e6, K_WKB(d_arr,5e-3)/(2*np.pi*1e6), color='k',lw=1.5)
ax.set_xlabel(r'$d$ (µm)'); ax.set_ylabel(r'$K_{\max}/2\pi$ (MHz)')
ax.set_title('(a)', loc='left',fontweight='bold')
ax.text(0.97,0.92,r'$\delta B=5\,$mT',transform=ax.transAxes,
        ha='right',fontsize=8)
ax.grid(True,which='both',alpha=0.2,ls=':',lw=0.5)

ax=ax1[0,1]
ax.semilogy(dB_arr*1e3, K_WKB(1e-6,dB_arr)/(2*np.pi*1e6), color='k',lw=1.5)
ax.set_xlabel(r'$\delta B$ (mT)'); ax.set_ylabel(r'$K_{\max}/2\pi$ (MHz)')
ax.set_title('(b)', loc='left',fontweight='bold')
ax.text(0.97,0.92,r'$d=1\,\mu$m',transform=ax.transAxes,ha='right',fontsize=8)
ax.grid(True,which='both',alpha=0.2,ls=':',lw=0.5)

ax=ax1[1,0]
ax.semilogx(K_arr/(2*np.pi*1e6), F_bound(K_arr), 'k',lw=1.5,
            label=r'$F_{\max}=1-\pi\Gamma/K$')
ax.axhline(0.99,color='red',ls='--',lw=1.0,label='$F=0.99$')
ax.axvline(K_star/(2*np.pi*1e6),color=C3,ls=':',lw=1.0,
           label=f'$K^*={K_star/(2*np.pi*1e6):.0f}$ MHz')
ax.set_xlabel(r'$K_{\max}/2\pi$ (MHz)'); ax.set_ylabel(r'$F_{\max}$')
ax.set_title('(c)',loc='left',fontweight='bold'); ax.set_ylim(0.60,1.02)
ax.legend(loc='lower right',fontsize=7.5)
ax.grid(True,which='both',alpha=0.2,ls=':',lw=0.5)

ax=ax1[1,1]
V_show = to_np(make_potential())
ax.plot(z_um, V_show/(2*np.pi*1e9),'k',lw=1.5)
ax.fill_between(z_um, V_show/(2*np.pi*1e9),0,
                where=np.abs(z_np)<0.5e-6,color=CB,alpha=0.5,label='barrier')
ax.set_xlabel(r'$z$ (µm)'); ax.set_ylabel(r'$V/2\pi$ (GHz)')
ax.set_title('(d)',loc='left',fontweight='bold')
ax.legend(loc='upper center',fontsize=8); ax.set_xlim(-22,22)
ax.grid(True,alpha=0.2,ls=':',lw=0.5)

fig1.suptitle('Analytical theory — WKB Josephson coupling and fidelity bound',
              fontsize=10,fontweight='bold')
fig1.savefig('fig1_theory.pdf'); fig1.savefig('fig1_theory.png',dpi=200)
print("  ✓ fig1_theory")

# ════════════════════════════════════════════════════════════════════════════
# 12. FIGURE 2 — GPE Dynamics
# ════════════════════════════════════════════════════════════════════════════
fig2=plt.figure(figsize=(7,8.5))
gs2=gridspec.GridSpec(3,4,figure=fig2,hspace=0.56,wspace=0.42)

panel_lbl=['(a)','(b)','(c)','(d)']
for k,(ctrl,tgt,lbl,eB) in enumerate(truth_cases):
    ax=fig2.add_subplot(gs2[0,k])
    s=sims[(ctrl,tgt)]
    nBp,nBm=integ(s,'B'); _,nAm=integ(s,'A')
    ax.plot(s['t_ns'],nBp,C1,lw=1.5,label=r'$N_{B+}$')
    ax.plot(s['t_ns'],nBm,C2,lw=1.5,label=r'$N_{B-}$')
    ax.plot(s['t_ns'],nAm,'k--',lw=0.8,alpha=0.7,label=r'$N_{A-}$')
    ax.set_xlabel('$t$ (ns)',labelpad=1); ax.set_ylabel('population',labelpad=1)
    ax.set_title(f'{panel_lbl[k]} {lbl}',fontsize=8,pad=2)
    ax.set_ylim(-0.05,1.12)
    if k==0: ax.legend(loc='center right',fontsize=6.5,framealpha=0.7)
    ax.grid(True,alpha=0.2,ls=':',lw=0.5)

# (e) density map
ax_d=fig2.add_subplot(gs2[1,:])
sf=sims[('1','0')]
dens=(sf['N_Bp']+sf['N_Bm']+sf['N_Ap']+sf['N_Am'])
ext=[sf['t_ns'][0],sf['t_ns'][-1],sf['z_um'][0],sf['z_um'][-1]]
im=ax_d.imshow(dens.T,origin='lower',aspect='auto',extent=ext,cmap='magma')
plt.colorbar(im,ax=ax_d,label=r'$\sum|\psi|^2$',pad=0.01,fraction=0.018)
ax_d.axhline( 0.5,color='cyan',ls='--',lw=0.8,alpha=0.8)
ax_d.axhline(-0.5,color='cyan',ls='--',lw=0.8,alpha=0.8)
ax_d.set_xlabel('$t$ (ns)'); ax_d.set_ylabel(r'$z$ (µm)')
ax_d.set_title(r'(e)  $|\psi(z,t)|^2$ — case $|10\rangle\!\to\!|11\rangle$',
               loc='left',fontweight='bold',fontsize=9)

# (f) Bloch
ax_bs=fig2.add_subplot(gs2[2,:2],projection='3d')
nBp_r,nBm_r=integ(sf,'B')
th=np.arccos(np.clip(nBp_r-nBm_r,-1,1))
ph=sf['phi_B']
bx=np.sin(th)*np.cos(ph); by=np.sin(th)*np.sin(ph); bz=np.cos(th)
u=np.linspace(0,2*np.pi,25); v=np.linspace(0,np.pi,25)
ax_bs.plot_wireframe(np.outer(np.cos(u),np.sin(v)),
                     np.outer(np.sin(u),np.sin(v)),
                     np.outer(np.ones_like(u),np.cos(v)),
                     color='gray',alpha=0.12,lw=0.3)
ax_bs.plot(bx,by,bz,C3,lw=1.8)
ax_bs.scatter([bx[0]],[by[0]],[bz[0]],color=C1,s=35,zorder=5)
ax_bs.scatter([bx[-1]],[by[-1]],[bz[-1]],color=C2,s=35,zorder=5)
ax_bs.text(0,0,1.25,r'$|0\rangle$',ha='center',fontsize=9,color=C1)
ax_bs.text(0,0,-1.32,r'$|1\rangle$',ha='center',fontsize=9,color=C2)
ax_bs.set_xlim(-1.3,1.3);ax_bs.set_ylim(-1.3,1.3);ax_bs.set_zlim(-1.3,1.3)
ax_bs.set_xlabel('x',labelpad=-4);ax_bs.set_ylabel('y',labelpad=-4)
ax_bs.set_zlabel('z',labelpad=-4)
ax_bs.set_title('(f) Bloch sphere',loc='left',fontweight='bold',fontsize=9)
ax_bs.tick_params(labelsize=7)

# (g) truth-table bar
ax_tt=fig2.add_subplot(gs2[2,2:])
cols=[C1,C1,C2,C2]
ax_tt.bar(range(4),tt_F,color=cols,edgecolor='k',lw=0.5)
ax_tt.set_xticks(range(4))
ax_tt.set_xticklabels([r'$|00\rangle$',r'$|01\rangle$',
                        r'$|10\rangle$',r'$|11\rangle$'])
ax_tt.axhline(0.99,color='red',ls='--',lw=1.0,label='$F=0.99$')
ax_tt.set_ylim(0,1.15); ax_tt.set_ylabel('Fidelity $F$')
ax_tt.set_title(f'(g) $\\bar{{F}}={np.mean(tt_F):.3f}$ (pumped)',
                loc='left',fontweight='bold',fontsize=9)
for k2,v in enumerate(tt_F):
    ax_tt.text(k2,v+0.03,f'{v:.2f}',ha='center',fontsize=8)
ax_tt.legend(fontsize=8); ax_tt.grid(True,axis='y',alpha=0.2,ls=':',lw=0.5)

fig2.suptitle('GPE truth-table simulation  '
              r'($K_{\max}/2\pi=250\,$MHz, $T_{\rm gate}=1.0\,$ns)',
              fontsize=10,fontweight='bold')
fig2.savefig('fig2_dynamics.pdf'); fig2.savefig('fig2_dynamics.png',dpi=200)
print("  ✓ fig2_dynamics")

# ════════════════════════════════════════════════════════════════════════════
# 13. FIGURE 3 — Phase diagrams + Universality
# ════════════════════════════════════════════════════════════════════════════
fig3,ax3=plt.subplots(1,3,figsize=(10,3.8))
fig3.subplots_adjust(wspace=0.40)

# (a) F(K,Γ)
ax=ax3[0]
im=ax.pcolormesh(G_sw/(2*np.pi*1e6),K_sw/(2*np.pi*1e6),F_KG,
                 shading='auto',cmap='RdYlGn',vmin=0.5,vmax=1.0)
cs=ax.contour(G_sw/(2*np.pi*1e6),K_sw/(2*np.pi*1e6),F_KG,
              levels=[0.90,0.95,0.99],colors='k',linewidths=0.8)
ax.clabel(cs,fmt='%.2f',fontsize=7)
ax.axhline(K_demo/(2*np.pi*1e6),color='red',ls='--',lw=1.0)
ax.axvline(Gamma0/(2*np.pi*1e6),color='red',ls='--',lw=1.0,
           label=r'YIG $\alpha{=}2{\times}10^{-4}$')
ax.set_xscale('log');ax.set_yscale('log')
ax.set_xlabel(r'$\Gamma/2\pi$ (MHz)'); ax.set_ylabel(r'$K_{\max}/2\pi$ (MHz)')
ax.set_title('(a)',loc='left',fontweight='bold')
ax.legend(fontsize=7,loc='upper left')
plt.colorbar(im,ax=ax,label='Fidelity $F$',pad=0.02)

# (b) F(d,dB)
ax=ax3[1]
im=ax.pcolormesh(dB_sw,d_sw,F_Kd,shading='auto',
                 cmap='RdYlGn',vmin=0.5,vmax=1.0)
cs=ax.contour(dB_sw,d_sw,F_Kd,
              levels=[0.90,0.95,0.99],colors='k',linewidths=0.8)
ax.clabel(cs,fmt='%.2f',fontsize=7)
ax.scatter([dB_sw[idx[1]]],[d_sw[idx[0]]],
           marker='*',s=150,color='k',zorder=5,
           label=f'd={d_sw[idx[0]]:.1f}µm\nδB={dB_sw[idx[1]]:.0f}mT')
ax.set_xlabel(r'$\delta B$ (mT)'); ax.set_ylabel(r'$d$ (µm)')
ax.set_title('(b)',loc='left',fontweight='bold')
ax.legend(fontsize=7,loc='upper right')
plt.colorbar(im,ax=ax,label='Fidelity $F$',pad=0.02)

# (c) Bell + CNOT inset
ax=ax3[2]
amps=np.abs(bell)**2
ax.bar(range(4),amps,color=[C1,'lightgray','lightgray',C1],
       edgecolor='k',lw=0.5)
ax.set_xticks(range(4))
ax.set_xticklabels([r'$|00\rangle$',r'$|01\rangle$',
                    r'$|10\rangle$',r'$|11\rangle$'])
ax.set_ylim(0,0.65); ax.set_ylabel('Probability')
ax.set_title(f'(c) Bell state  $F={bell_F:.4f}$',
             loc='left',fontweight='bold',fontsize=9)
for k2,v in enumerate(amps):
    if v>0.01: ax.text(k2,v+0.02,f'{v:.3f}',ha='center',fontsize=8)
ax_in=ax.inset_axes([0.20,0.25,0.50,0.60])
ax_in.imshow(np.abs(CNOT_mat)**2,cmap='Blues',vmin=0,vmax=1,aspect='equal')
labs=[r'$|00\rangle$',r'$|01\rangle$',r'$|10\rangle$',r'$|11\rangle$']
ax_in.set_xticks(range(4));ax_in.set_yticks(range(4))
ax_in.set_xticklabels(labs,fontsize=5.5)
ax_in.set_yticklabels(labs,fontsize=5.5)
ax_in.set_title(r'$|U|^2$',fontsize=7)

fig3.suptitle('Parameter sweeps and universality',fontsize=10,fontweight='bold')
fig3.savefig('fig3_sweep.pdf'); fig3.savefig('fig3_sweep.png',dpi=200)
print("  ✓ fig3_sweep")

# ════════════════════════════════════════════════════════════════════════════
# 14. FIGURE 4 — Ensemble fidelity
# ════════════════════════════════════════════════════════════════════════════
fig4,ax4=plt.subplots(1,3,figsize=(10,3.5))
fig4.subplots_adjust(wspace=0.38)

ax=ax4[0]
ax.plot(noise_sw, F_n_sw,'ko-',ms=5,lw=1.5)
ax.axvline(sig_th,color='red',ls='--',lw=1.0,
           label=f'300 K noise\n$\\sigma_{{th}}={sig_th:.4f}$')
ax.axhline(0.99,color=C3,ls=':',lw=1.0,label='$F=0.99$')
ax.set_xlabel(r'Noise amplitude $\sigma$'); ax.set_ylabel('Ensemble $F$')
ax.set_title('(a)',loc='left',fontweight='bold')
ax.set_ylim(0.75,1.02); ax.legend(fontsize=7.5)
ax.grid(True,alpha=0.2,ls=':',lw=0.5)

ax=ax4[1]
ax.semilogx(K_ens_sw/(2*np.pi*1e6),F_K_sw,'ko-',ms=4,lw=1.5,
            label='Ensemble (N=200)')
Ka=np.logspace(7.5,9.5,200)
ax.semilogx(Ka/(2*np.pi*1e6),F_bound(Ka),'b--',lw=1.2,
            label=r'$1-\pi\Gamma/K$')
ax.axvline(K_demo/(2*np.pi*1e6),color='red',ls='--',lw=1.0,label='Op. pt.')
ax.axhline(0.99,color=C3,ls=':',lw=1.0)
ax.set_xlabel(r'$K_{\max}/2\pi$ (MHz)'); ax.set_ylabel('Fidelity $F$')
ax.set_title('(b)',loc='left',fontweight='bold')
ax.set_ylim(0.5,1.02); ax.legend(fontsize=7.0)
ax.grid(True,which='both',alpha=0.2,ls=':',lw=0.5)

ax=ax4[2]
ax.bar(range(4),F_ens,color=[C1,C1,C2,C2],
       yerr=F_err,capsize=4,edgecolor='k',lw=0.5,
       error_kw={'lw':1.0})
ax.set_xticks(range(4))
ax.set_xticklabels([r'$|00\rangle$',r'$|01\rangle$',
                    r'$|10\rangle$',r'$|11\rangle$'])
ax.axhline(0.99,color='red',ls='--',lw=1.0,label='$F=0.99$')
ax.set_ylim(0.80,1.06); ax.set_ylabel('Fidelity $F$')
ax.set_title(f'(c) $\\bar{{F}}={F_ens_mean:.4f}\\pm{F_ens_std:.4f}$',
             loc='left',fontweight='bold',fontsize=9)
for k2,(f,e) in enumerate(zip(F_ens,F_err)):
    ax.text(k2,f+e+0.004,f'{f:.3f}',ha='center',fontsize=7.5)
ax.legend(fontsize=8); ax.grid(True,axis='y',alpha=0.2,ls=':',lw=0.5)

fig4.suptitle(f'Ensemble fidelity — 500 trajectories, $T=300\\,$K, no renormalisation',
              fontsize=10,fontweight='bold')
fig4.savefig('fig4_ensemble.pdf'); fig4.savefig('fig4_ensemble.png',dpi=200)
print("  ✓ fig4_ensemble")

# ════════════════════════════════════════════════════════════════════════════
# 15. SAVE TO DRIVE + SUMMARY
# ════════════════════════════════════════════════════════════════════════════
# Uncomment if running on Colab with Drive mounted:
# import shutil
# DRIVE = '/content/drive/MyDrive/magnon_cnot'
# os.makedirs(DRIVE, exist_ok=True)
# for f in ['fig1_theory.pdf','fig2_dynamics.pdf',
#           'fig3_sweep.pdf','fig4_ensemble.pdf']:
#     shutil.copy(f, f'{DRIVE}/{f}')

summary = f"""
╔══════════════════════════════════════════════════════════════════╗
║  magnon-BEC CNOT  —  Publication summary                        ║
╠══════════════════════════════════════════════════════════════════╣
║  Part 1: Analytical                                             ║
║    K*/2π  = {K_star/(2*np.pi)/1e6:6.1f} MHz   (F≥0.99 threshold)            ║
║    τ_gate = {tau_gate(K_demo)*1e9:6.2f} ns    at K/2π={K_demo/(2*np.pi)/1e6:.0f} MHz           ║
║                                                                 ║
║  Part 2: GPE truth-table  (pumped model)                        ║
║    F(00→00) = {tt_F[0]:.4f}                                       ║
║    F(01→01) = {tt_F[1]:.4f}                                       ║
║    F(10→11) = {tt_F[2]:.4f}                                       ║
║    F(11→10) = {tt_F[3]:.4f}                                       ║
║    F_avg   = {np.mean(tt_F):.4f}                                       ║
║                                                                 ║
║  Part 4: Ensemble (500 traj, 300K, no pump)                     ║
║    F_avg = {F_ens_mean:.4f} ± {F_ens_std:.4f}                              ║
║    σ_thermal = {sig_th:.5f}                                  ║
║    Analytical bound at K_demo: {F_bound(K_demo):.4f}                   ║
║                                                                 ║
║  Part 5: Universality                                           ║
║    H overlap = {H_ov:.4f}                                        ║
║    Bell F    = {bell_F:.4f}                                        ║
║    Gate set  = {{CNOT, H, S, T}}  →  BQP-complete               ║
║                                                                 ║
║  Figures: fig1_theory / fig2_dynamics / fig3_sweep / fig4_ensemble  ║
║                                                                 ║
║  LaTeX line for abstract:                                       ║
║    \\bar{{F}} = {F_ens_mean:.3f} \\pm {F_ens_std:.3f} \\quad (T=300\\,\\mathrm{{K}})    ║
╚══════════════════════════════════════════════════════════════════╝
"""
print(summary)

with open('magnon_cnot_summary.txt','w') as f: f.write(summary)

# Download on Colab:
# from google.colab import files
# for fn in ['fig1_theory.pdf','fig2_dynamics.pdf',
#            'fig3_sweep.pdf','fig4_ensemble.pdf',
#            'magnon_cnot_summary.txt']:
#     files.download(fn)
