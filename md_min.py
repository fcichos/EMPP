import numpy as np
import matplotlib.pyplot as plt

class Atom:
    def __init__(self, position, mass=1.0):
        self.position = position
        self.velocity = np.random.randn(2) * 0.1  # Small random initial velocity
        self.force = np.zeros(2)
        self.mass = mass

    def update_position(self, dt):
        self.position += self.velocity * dt + 0.5 * self.force/self.mass * dt**2

    def update_velocity(self, dt, new_force):
        self.velocity += 0.5 * (new_force + self.force)/self.mass * dt
        self.force = new_force

    def apply_periodic_boundaries(self, box_size):
        self.position = self.position % box_size

class MDSimulation:
    def __init__(self, atoms, box_size, dt):
        self.atoms = atoms
        self.box_size = np.array(box_size)
        self.dt = dt

    def minimum_image_distance(self, pos1, pos2):
        delta = pos1 - pos2
        delta = delta - self.box_size * np.round(delta / self.box_size)
        return delta

    def calculate_forces(self):
        # Reset forces
        for atom in self.atoms:
            atom.force = np.zeros(2)

        # Calculate forces between all pairs
        for i, atom1 in enumerate(self.atoms):
            for atom2 in self.atoms[i+1:]:
                r = self.minimum_image_distance(atom1.position, atom2.position)
                r_mag = np.linalg.norm(r)

                # Simple Lennard-Jones force
                force_mag = 24 * (2/r_mag**13 - 1/r_mag**7)
                force = force_mag * r/r_mag

                atom1.force += force
                atom2.force -= force

    def step(self):
        # Update positions
        for atom in self.atoms:
            atom.update_position(self.dt)
            atom.apply_periodic_boundaries(self.box_size)

        # Calculate new forces
        self.calculate_forces()

        # Update velocities
        for atom in self.atoms:
            atom.update_velocity(self.dt, atom.force)

# Create simulation
box_size = np.array([10.0, 10.0])
n_atoms = 50

# Create atoms in a grid
atoms = []
n = int(np.sqrt(n_atoms))
spacing = box_size[0] / n
for i in range(n_atoms):
    pos = np.array([spacing * (i % n), spacing * (i // n)])
    atoms.append(Atom(pos))

# Create simulation
sim = MDSimulation(atoms, box_size, dt=0.001)

# Run simulation with visualization
fig, ax = plt.subplots()
for step in range(1000):
    sim.step()

    # Plot
    if step % 10 == 0:  # Update plot every 10 steps
        ax.clear()
        positions = np.array([atom.position for atom in atoms])
        ax.scatter(positions[:,0], positions[:,1])
        ax.set_xlim(0, box_size[0])
        ax.set_ylim(0, box_size[1])
        plt.pause(0.01)

plt.show()
