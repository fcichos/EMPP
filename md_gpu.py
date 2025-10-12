# %% import modules
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import matplotlib.patches as patches

# %% Make sure to use GPU if available
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# %% Define the Atom class
print(device)


# %% Define the Atom class
class AtomSystem:
    def __init__(self, positions, velocities, masses, types, box_size):
        self.positions = positions.to(device)
        self.velocities = velocities.to(device)
        self.masses = masses.to(device)
        self.types = types
        self.forces = torch.zeros_like(positions).to(device)
        self.box_size = torch.tensor(box_size).to(device)

    def reset_forces(self):
        self.forces.zero_()

    def apply_periodic_boundaries(self):
        self.positions = self.positions % self.box_size


# %% Define the ForceField class
class ForceField:
    def __init__(self):
        self.parameters = {
            'C': {'epsilon': 1.615, 'sigma': 1.36},
            'H': {'epsilon': 1.0, 'sigma': 1.0},
            'O': {'epsilon': 1.846, 'sigma': 3.0},
        }

    def calculate_forces(self, system):
        n_atoms = len(system.positions)

        # Create position differences matrix
        pos_i = system.positions.unsqueeze(0)
        pos_j = system.positions.unsqueeze(1)

        # Calculate distances with periodic boundary conditions
        diff = pos_i - pos_j
        diff = diff - system.box_size * torch.round(diff / system.box_size)

        # Calculate distances
        r = torch.norm(diff, dim=2)
        r = r.masked_fill(r == 0, float('inf'))  # Avoid self-interactions

        # Get parameters for all pairs
        sigma = torch.tensor(self.parameters[system.types[0]]['sigma']).to(device)
        epsilon = torch.tensor(self.parameters[system.types[0]]['epsilon']).to(device)

        # Calculate forces
        sr6 = (sigma/r)**6
        sr7 = sr6/r  # This was missing
        sr12 = sr6**2
        # Apply cutoff
        mask = (r < 3.5*sigma).float()

        force_magnitude = 24 * epsilon * (2*sr12 - sr7) * mask

        # Calculate force vectors
        forces = force_magnitude.unsqueeze(-1) * (diff / r.unsqueeze(-1))

        # Sum forces on each atom
        system.forces = forces.sum(dim=1)

class MDSimulation:
    def __init__(self, system, forcefield, timestep, temperature=5.0):
        self.system = system
        self.forcefield = forcefield
        self.timestep = timestep
        self.temperature = temperature
        self.kB = 1.0  # Boltzmann constant in reduced units

    def apply_thermostat(self):
        current_temp = torch.mean(self.system.masses.unsqueeze(1) *
                                torch.sum(self.system.velocities**2, dim=1)) / (2 * self.kB)
        # Move to CPU before numpy operations
        scaling_factor = torch.sqrt(torch.tensor(self.temperature / current_temp.cpu()))
        # Move scaling factor back to the same device as velocities
        scaling_factor = scaling_factor.to(self.system.velocities.device)
        self.system.velocities *= scaling_factor

    def step(self):
        dt = self.timestep

        # Update positions
        self.system.positions += (self.system.velocities * dt +
                                0.5 * (self.system.forces/self.system.masses.unsqueeze(1)) * dt**2)

        self.system.apply_periodic_boundaries()

        # Calculate new forces
        old_forces = self.system.forces.clone()
        self.system.reset_forces()
        self.forcefield.calculate_forces(self.system)

        # Update velocities
        self.system.velocities += 0.5 * (self.system.forces + old_forces) / self.system.masses.unsqueeze(1) * dt

        # Apply thermostat every few steps if needed
        if step % 10 == 0:
            self.apply_thermostat()

def initialize_system(num_atoms, box_size, temperature, atom_type="H"):
    # Initialize positions on a grid
    n = int(np.ceil(np.sqrt(num_atoms)))
    spacing = min(box_size) / n

    positions = []
    for i in range(num_atoms):
        row = i // n
        col = i % n
        pos = [col * spacing + spacing/2, row * spacing + spacing/2]
        positions.append(pos)

    positions = torch.tensor(positions, dtype=torch.float32)

    # Set masses
    masses = torch.ones(num_atoms)

    # Initialize velocities from Maxwell-Boltzmann distribution
    kB = 1.0  # Boltzmann constant in reduced units
    velocities = torch.randn(num_atoms, 2) * np.sqrt(kB * temperature / masses.unsqueeze(1))

    # Remove center of mass motion
    com_velocity = torch.mean(velocities, dim=0)
    velocities -= com_velocity

    types = [atom_type] * num_atoms

    return AtomSystem(positions, velocities, masses, types, box_size)



# Run simulation
box_size = [50.0, 50.0]
num_atoms = 200
T = 5
dt = 0.001

system = initialize_system(num_atoms, box_size, T)
ff = ForceField()
sim = MDSimulation(system, ff, dt, temperature=T)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

for step in range(1000):
    clear_output(wait=True)

    sim.step()

    # Get positions for plotting (move to CPU)
    positions = system.positions.cpu().numpy()

    circle = patches.Circle((positions[0,0], positions[0,1]),
                          ff.parameters[system.types[0]]["sigma"],
                          edgecolor="white", fill=False)
    ax.add_patch(circle)
    ax.scatter(positions[:,0], positions[:,1], color="red")
    ax.set_xlim(0, box_size[0])
    ax.set_ylim(0, box_size[1])
    ax.axis("off")

    display(fig)
    ax.clear()
