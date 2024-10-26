import random

import torch as t

from config import device
from camera import Camera
from hittable import HittableList
from materials import Dielectric, Lambertian, Metal
from sphere import Sphere

# Choose device
print(f"Using device: {device}")


def random_double(min_val=0.0, max_val=1.0):
    return min_val + (max_val - min_val) * random.random()


def random_color():
    return t.tensor([random.random(), random.random(), random.random()], device=device)


# Initialize the world
world = HittableList()

# Ground
ground_material = Lambertian(t.tensor([0.5, 0.5, 0.5], device=device))
world.add(Sphere(t.tensor([0, -1000, 0], device=device), 1000, ground_material))

# Random small spheres
for a in range(-11, 11):
    for b in range(-11, 11):
        choose_mat = random_double()
        center = t.tensor([a + 0.9 * random_double(), 0.2, b + 0.9 * random_double()], device=device)

        if (center - t.tensor([4, 0.2, 0], device=device)).norm() > 0.9:
            if choose_mat < 0.8:
                # Diffuse
                albedo = random_color() * random_color()
                sphere_material = Lambertian(albedo)
            elif choose_mat < 0.95:
                # Metal
                albedo = random_color() * 0.5 + 0.5
                fuzz = random_double(0, 0.5)
                sphere_material = Metal(albedo, fuzz)
            else:
                # Glass
                sphere_material = Dielectric(1.5)

            world.add(Sphere(center, 0.2, sphere_material))

# Three larger spheres
material1 = Dielectric(1.5)
world.add(Sphere(t.tensor([0, 1, 0], device=device), 1.0, material1))

material2 = Lambertian(t.tensor([0.4, 0.2, 0.1], device=device))
world.add(Sphere(t.tensor([-4, 1, 0], device=device), 1.0, material2))

material3 = Metal(t.tensor([0.7, 0.6, 0.5], device=device), 0.0)
world.add(Sphere(t.tensor([4, 1, 0], device=device), 1.0, material3))

# Initialize the camera
camera = Camera(
    image_width=400,
    samples_per_pixel=10,
    aspect_ratio=16.0 / 9.0,
    max_depth=10,
    vfov=20,
    look_from=t.tensor([13, 2, 3], dtype=t.float32, device=device),
    look_at=t.tensor([0, 0, 0], dtype=t.float32, device=device),
    vup=t.tensor([0, 1, 0], dtype=t.float32, device=device),
    defocus_angle=0.6,
    focus_dist=10.0,
)


# Render the image
image = camera.render(world)
image.save("image.png")
image.show()
