# wave_double_slit.py
# -------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.fft import fft2, ifft2, fftshift

# optional: speed up the loop
try:
    from numba import jit

    _jit = jit
except ImportError:
    _jit = lambda f: f

# ------------------------------------------------------------------
# PARAMETERS ---------------------------------------------------------
Lx, Ly = 200.0, 200.0  # physical size (units of ℏ/m)
Nx, Ny = 512, 512  # grid points
dx, dy = Lx / Nx, Ly / Ny
x = np.linspace(-Lx / 2, Lx / 2, Nx)
y = np.linspace(-Ly / 2, Ly / 2, Ny)
X, Y = np.meshgrid(x, y, indexing="ij")

# Momentum space grid
kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
KX, KY = np.meshgrid(kx, ky, indexing="ij")
K2 = KX**2 + KY**2

dt = 0.2  # time step
nsteps = 200  # number of time steps

# --------------------------------------------------------------
# Initial wavepacket ------------------------------------------------
x0, y0 = 0.0, -80.0  # launch point
kx0, ky0 = 0.0, 1.5  # average momentum (toward +y)
sigma = 10.0  # width of Gaussian
psi0 = np.exp(
    -((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2)
    + 1j * (kx0 * (X - x0) + ky0 * (Y - y0))
)

# --------------------------------------------------------------
# Double‑slit potential ------------------------------------------
# We simply make two narrow strips of *zero* potential in a wall
slit_width = 5.0
slit_sep = 10.0
wall_y = -30.0
wall_thick = 2.0

# Start with free space (V = 0 everywhere)
V = np.zeros_like(X)

# Create wall at wall_y extending over entire x-range
wall_mask = (Y > wall_y - wall_thick / 2) & (Y < wall_y + wall_thick / 2)
V[wall_mask] = 1e3  # high potential wall

# Open two slits of width 5, separated by 10 (centers at x = ±5)
slit_mask = (
    (np.abs(X - slit_sep / 2) < slit_width / 2)  # slit at x = +5
    | (np.abs(X + slit_sep / 2) < slit_width / 2)  # slit at x = -5
)
V[wall_mask & slit_mask] = 0.0  # open the slits


# --------------------------------------------------------------
# Pre‑compute operators ------------------------------------------
expV = np.exp(-1j * V * dt)  # potential kick
expK = np.exp(-1j * K2 * dt / 2.0)  # kinetic kick


# --------------------------------------------------------------
# @_jit(nopython=False, parallel=True)
def time_step(psi):
    """One SSFM step: potential -> kinetic -> potential"""
    # potential kick (real space)
    psi = psi * expV
    # kinetic kick (Fourier space)
    psi_k = fft2(psi)
    psi_k = psi_k * expK
    psi = ifft2(psi_k)
    # second potential kick
    psi = psi * expV
    return psi


# --------------------------------------------------------------
# Run animation ---------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.imshow(
    np.abs(psi0) ** 2,
    extent=[x.min(), x.max(), y.min(), y.max()],
    cmap="inferno",
    origin="lower",
    vmin=0,
    vmax=1.0,
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Quantum double‑slit (real‑time)")


def init():
    cax.set_array(np.abs(psi0) ** 2)

    return (cax,)


psi = psi0.copy()


def animate(i):
    global psi
    psi = time_step(psi)
    img = np.abs(psi) ** 2
    cax.set_array(img)
    ax.set_title(f"Time step {i + 1}/{nsteps}")
    return (cax,)


ani = animation.FuncAnimation(
    fig, animate, frames=nsteps, init_func=init, blit=True, interval=30
)
plt.show()
