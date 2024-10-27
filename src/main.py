import random

import torch as t
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

from camera import Camera
from config import device
from materials import MaterialType
from sphere import SphereList

# Choose device
print(f"Using device: {device}")


@jaxtyped(typechecker=typechecker)
def random_double(min_val=0.0, max_val=1.0):
    return min_val + (max_val - min_val) * random.random()


@jaxtyped(typechecker=typechecker)
def random_color():
    return t.tensor([random.random(), random.random(), random.random()], device=device)


# Initialize lists to collect sphere data
sphere_centers = []
sphere_radii = []
material_types = []
albedos = []
fuzzes = []
refractive_indices = []

# Ground sphere
sphere_centers.append(t.tensor([0, -1000, 0], device=device))
sphere_radii.append(1000.0)
material_types.append(MaterialType.Lambertian)
albedos.append(t.tensor([0.5, 0.5, 0.5], device=device))
fuzzes.append(0.0)
refractive_indices.append(0.0)

# Random small spheres
for a in range(-11, 11):
    for b in range(-11, 11):
        choose_mat = random_double()
        center = t.tensor([a + 0.9 * random_double(), 0.2, b + 0.9 * random_double()], device=device)
        if (center - t.tensor([4, 0.2, 0], device=device)).norm() > 0.9:
            if choose_mat < 0.8:
                # Diffuse
                albedo = random_color() * random_color()
                material_type = MaterialType.Lambertian
                fuzz = 0.0
                refractive_index = 0.0
            elif choose_mat < 0.95:
                # Metal
                albedo = random_color() * 0.5 + 0.5
                fuzz = random_double(0, 0.5)
                material_type = MaterialType.Metal
                refractive_index = 0.0
            else:
                # Glass
                albedo = t.tensor([0.0, 0.0, 0.0], device=device)
                fuzz = 0.0
                refractive_index = 1.5
                material_type = MaterialType.Dielectric
            sphere_centers.append(center)
            sphere_radii.append(0.2)
            material_types.append(material_type)
            albedos.append(albedo)
            fuzzes.append(fuzz)
            refractive_indices.append(refractive_index)

# Three larger spheres
# Sphere 1
sphere_centers.append(t.tensor([0, 1, 0], device=device))
sphere_radii.append(1.0)
material_types.append(MaterialType.Dielectric)
albedos.append(t.tensor([0.0, 0.0, 0.0], device=device))
fuzzes.append(0.0)
refractive_indices.append(1.5)

# Sphere 2
sphere_centers.append(t.tensor([-4, 1, 0], device=device))
sphere_radii.append(1.0)
material_types.append(MaterialType.Lambertian)
albedos.append(t.tensor([0.4, 0.2, 0.1], device=device))
fuzzes.append(0.0)
refractive_indices.append(0.0)

# Sphere 3
sphere_centers.append(t.tensor([4, 1, 0], device=device))
sphere_radii.append(1.0)
material_types.append(MaterialType.Metal)
albedos.append(t.tensor([0.7, 0.6, 0.5], device=device))
fuzzes.append(0.0)
refractive_indices.append(0.0)

# Convert lists to tensors
sphere_centers = t.stack(sphere_centers)  # [M, 3]
sphere_radii = t.tensor(sphere_radii, device=device)  # [M]
material_types = t.tensor(material_types, device=device)  # [M]
albedos = t.stack(albedos)  # [M, 3]
fuzzes = t.tensor(fuzzes, device=device)  # [M]
refractive_indices = t.tensor(refractive_indices, device=device)  # [M]

# Create the SphereList
world = SphereList(
    centers=sphere_centers,
    radii=sphere_radii,
    material_types=material_types,
    albedos=albedos,
    fuzzes=fuzzes,
    refractive_indices=refractive_indices,
)


# Initialize the camera
camera = Camera(
    image_width=1920,
    samples_per_pixel=120,
    aspect_ratio=16.0 / 9.0,
    max_depth=50,
    vfov=20,
    look_from=t.tensor([13, 2, 3], dtype=t.float32, device=device),
    look_at=t.tensor([0, 0, 0], dtype=t.float32, device=device),
    vup=t.tensor([0, 1, 0], dtype=t.float32, device=device),
    defocus_angle=0.6,
    focus_dist=10.0,
    batch_size=10_000,
)

# Render the image
image = camera.render(world)
image.save("image.png")
image.show()
