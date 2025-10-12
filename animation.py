# %% Cell 4
#
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from scipy.spatial.distance import cdist

n_side = 25

x = np.linspace(0.05, 0.95, n_side)
y = np.linspace(0.05, 0.95, n_side)
xx, yy = np.meshgrid(x, y)
particles = np.vstack([xx.ravel(), yy.ravel()]).T

velocities = np.random.normal(scale=0.005, size=(n_side**2, 2))

radius = 0.0177
fig, ax = plt.subplots(figsize=(9,9))

n_steps = 200

for _ in range(n_steps):
    clear_output(wait=True)

    # Update particle positions based on their velocities
    particles += velocities
    # Apply periodic boundary conditions in x direction (wrap around at 0 and 1)
    particles[:, 0] = particles[:, 0] % 1
    # Apply periodic boundary conditions in y direction (wrap around at 0 and 1)
    particles[:, 1] = particles[:, 1] % 1
    # Calculate distances between all pairs of particles
    distances = cdist(particles, particles)

    collisions = np.triu(distances < 2*radius, 1)

    for i, j in zip(*np.nonzero(collisions)):
        velocities[i], velocities[j] = velocities[j], velocities[i].copy()

        overlap = 2*radius - distances[i, j]
        direction = particles[i] - particles[j]
        direction /= np.linalg.norm(direction)
        particles[i] += 0.5 * overlap * direction
        particles[j] -= 0.5 * overlap * direction

    ax.scatter(particles[:, 0], particles[:, 1], s=100, edgecolors='r', facecolors='none')
    #ax.quiver(particles[:, 0], particles[:, 1], velocities[:, 0], velocities[:, 1], color='b')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    display(fig)
    plt.pause(0.01)
    ax.clear()
