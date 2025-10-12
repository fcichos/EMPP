


```{python}
# | echo: false
import numpy as np

def initialize_reduced_velocities(N, T_reduced):
    """
    Initialize velocities for N particles at reduced temperature T*
    following the Maxwell-Boltzmann distribution in reduced units.

    Parameters:
    N (int): Number of particles
    T_reduced (float): Reduced temperature T* = kT/ε

    Returns:
    ndarray: Array of reduced velocities (N x 3)
    """
    # Generate random velocities from a normal distribution
    # Standard deviation is sqrt(T*) since mass = 1 in reduced units
    velocities = np.random.normal(0, np.sqrt(T_reduced), (N, 3))

    # Remove center of mass motion
    velocities -= np.mean(velocities, axis=0)

    # Scale velocities to match desired temperature
    # In reduced units, mass = 1
    current_temp = np.sum(velocities**2) / (3 * N)
    velocities *= np.sqrt(T_reduced/current_temp)

    return velocities

def convert_to_real_units(v_reduced, epsilon, sigma, mass):
    """
    Convert reduced velocities to real units.

    Parameters:
    v_reduced: Velocities in reduced units
    epsilon: LJ well depth (J)
    sigma: LJ length parameter (m)
    mass: Particle mass (kg)

    Returns:
    ndarray: Velocities in m/s
    """
    conversion_factor = np.sqrt(epsilon/mass)
    return v_reduced * conversion_factor

# Example usage
N = 1000  # number of particles
T_reduced = 1.0  # typical reduced temperature

# Initialize velocities in reduced units
v_reduced = initialize_reduced_velocities(N, T_reduced)

# Convert to real units (example values for Argon)
epsilon = 1.65e-21  # J
sigma = 3.4e-10    # m
mass = 6.63e-26    # kg

v_real = convert_to_real_units(v_reduced, epsilon, sigma, mass)
```
