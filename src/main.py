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


def create_random_spheres_scene():
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
    sphere_centers.append(t.tensor([0, 1, 0], device=device))
    sphere_radii.append(1.0)
    material_types.append(MaterialType.Dielectric)
    albedos.append(t.tensor([0.0, 0.0, 0.0], device=device))
    fuzzes.append(0.0)
    refractive_indices.append(1.5)

    sphere_centers.append(t.tensor([-4, 1, 0], device=device))
    sphere_radii.append(1.0)
    material_types.append(MaterialType.Lambertian)
    albedos.append(t.tensor([0.4, 0.2, 0.1], device=device))
    fuzzes.append(0.0)
    refractive_indices.append(0.0)

    sphere_centers.append(t.tensor([4, 1, 0], device=device))
    sphere_radii.append(1.0)
    material_types.append(MaterialType.Metal)
    albedos.append(t.tensor([0.7, 0.6, 0.5], device=device))
    fuzzes.append(0.0)
    refractive_indices.append(0.0)

    # Convert lists to tensors
    sphere_centers = t.stack(sphere_centers)
    sphere_radii = t.tensor(sphere_radii, device=device)
    material_types = t.tensor(material_types, device=device)
    albedos = t.stack(albedos)
    fuzzes = t.tensor(fuzzes, device=device)
    refractive_indices = t.tensor(refractive_indices, device=device)

    world = SphereList(
        centers=sphere_centers,
        radii=sphere_radii,
        material_types=material_types,
        albedos=albedos,
        fuzzes=fuzzes,
        refractive_indices=refractive_indices,
    )

    camera = Camera(
        image_width=1080,
        samples_per_pixel=50,
        aspect_ratio=16.0 / 9.0,
        max_depth=50,
        vfov=20,
        look_from=t.tensor([13, 2, 3], dtype=t.float32, device=device),
        look_at=t.tensor([0, 0, 0], dtype=t.float32, device=device),
        vup=t.tensor([0, 1, 0], dtype=t.float32, device=device),
        defocus_angle=0.6,
        focus_dist=10.0,
        batch_size=50_000,
    )

    return world, camera


def create_material_showcase_scene():
    sphere_centers = []
    sphere_radii = []
    material_types = []
    albedos = []
    fuzzes = []
    refractive_indices = []

    # Ground sphere
    sphere_centers.append(t.tensor([0, -100.5, -1], device=device))
    sphere_radii.append(100)
    material_types.append(MaterialType.Lambertian)
    albedos.append(t.tensor([0.5, 0.8, 0.3], device=device))
    fuzzes.append(0.0)
    refractive_indices.append(0.0)

    # Center glass sphere
    sphere_centers.append(t.tensor([0, 0, -1], device=device))
    sphere_radii.append(0.5)
    material_types.append(MaterialType.Dielectric)
    albedos.append(t.tensor([1.0, 1.0, 1.0], device=device))
    fuzzes.append(0.0)
    refractive_indices.append(1.5)

    # Left metallic sphere
    sphere_centers.append(t.tensor([-1.0, 0, -0.8], device=device))
    sphere_radii.append(0.4)
    material_types.append(MaterialType.Metal)
    albedos.append(t.tensor([0.8, 0.6, 0.2], device=device))
    fuzzes.append(0.2)
    refractive_indices.append(0.0)

    # Right matte sphere
    sphere_centers.append(t.tensor([1.0, -0.1, -0.7], device=device))
    sphere_radii.append(0.3)
    material_types.append(MaterialType.Lambertian)
    albedos.append(t.tensor([0.7, 0.3, 0.3], device=device))
    fuzzes.append(0.0)
    refractive_indices.append(0.0)

    # Small floating glass sphere
    sphere_centers.append(t.tensor([0.3, 0.3, -0.5], device=device))
    sphere_radii.append(0.15)
    material_types.append(MaterialType.Dielectric)
    albedos.append(t.tensor([1.0, 1.0, 1.0], device=device))
    fuzzes.append(0.0)
    refractive_indices.append(1.5)

    # Convert lists to tensors
    sphere_centers = t.stack(sphere_centers)
    sphere_radii = t.tensor(sphere_radii, device=device)
    material_types = t.tensor(material_types, device=device)
    albedos = t.stack(albedos)
    fuzzes = t.tensor(fuzzes, device=device)
    refractive_indices = t.tensor(refractive_indices, device=device)

    world = SphereList(
        centers=sphere_centers,
        radii=sphere_radii,
        material_types=material_types,
        albedos=albedos,
        fuzzes=fuzzes,
        refractive_indices=refractive_indices,
    )

    camera = Camera(
        image_width=1080,
        samples_per_pixel=50,
        aspect_ratio=16.0 / 9.0,
        max_depth=50,
        vfov=90,
        look_from=t.tensor([0, 0, 0], dtype=t.float32, device=device),
        look_at=t.tensor([0, 0, -1], dtype=t.float32, device=device),
        vup=t.tensor([0, 1, 0], dtype=t.float32, device=device),
        defocus_angle=0.0,
        focus_dist=1.0,
        batch_size=50_000,
    )

    return world, camera


def create_cornell_box_scene(max_depth):
    sphere_centers = []
    sphere_radii = []
    material_types = []
    albedos = []
    fuzzes = []
    refractive_indices = []

    # Red wall (left)
    sphere_centers.append(t.tensor([-101, 0, 0], device=device))
    sphere_radii.append(100)
    material_types.append(MaterialType.Lambertian)
    albedos.append(t.tensor([0.75, 0.25, 0.25], device=device))
    fuzzes.append(0.0)
    refractive_indices.append(0.0)

    # Green wall (right)
    sphere_centers.append(t.tensor([101, 0, 0], device=device))
    sphere_radii.append(100)
    material_types.append(MaterialType.Lambertian)
    albedos.append(t.tensor([0.25, 0.75, 0.25], device=device))
    fuzzes.append(0.0)
    refractive_indices.append(0.0)

    # White walls (top, bottom, back)
    for pos in [(0, 101, 0), (0, -101, 0), (0, 0, -101)]:
        sphere_centers.append(t.tensor(pos, device=device))
        sphere_radii.append(100)
        material_types.append(MaterialType.Lambertian)
        albedos.append(t.tensor([0.75, 0.75, 0.75], device=device))
        fuzzes.append(0.0)
        refractive_indices.append(0.0)

    # Add two spheres for more interesting scene
    # Glass sphere
    sphere_centers.append(t.tensor([-0.5, -0.7, -0.5], device=device))
    sphere_radii.append(0.3)
    material_types.append(MaterialType.Dielectric)
    albedos.append(t.tensor([1.0, 1.0, 1.0], device=device))
    fuzzes.append(0.0)
    refractive_indices.append(1.5)

    # Metal sphere
    sphere_centers.append(t.tensor([0.5, -0.7, 0.5], device=device))
    sphere_radii.append(0.3)
    material_types.append(MaterialType.Metal)
    albedos.append(t.tensor([0.8, 0.8, 0.8], device=device))
    fuzzes.append(0.1)
    refractive_indices.append(0.0)

    # Convert lists to tensors
    sphere_centers = t.stack(sphere_centers)
    sphere_radii = t.tensor(sphere_radii, device=device)
    material_types = t.tensor(material_types, device=device)
    albedos = t.stack(albedos)
    fuzzes = t.tensor(fuzzes, device=device)
    refractive_indices = t.tensor(refractive_indices, device=device)

    world = SphereList(
        centers=sphere_centers,
        radii=sphere_radii,
        material_types=material_types,
        albedos=albedos,
        fuzzes=fuzzes,
        refractive_indices=refractive_indices,
    )

    camera = Camera(
        image_width=1080,
        samples_per_pixel=20,
        aspect_ratio=1.0,  # Square aspect ratio for traditional Cornell box
        max_depth=max_depth,
        vfov=40,  # Narrower field of view
        look_from=t.tensor([0, 0, 4.5], dtype=t.float32, device=device),
        look_at=t.tensor([0, 0, 0], dtype=t.float32, device=device),
        vup=t.tensor([0, 1, 0], dtype=t.float32, device=device),
        defocus_angle=0.0,
        focus_dist=4.5,
        batch_size=50_000,
    )

    return world, camera


# Define scenes to render
scenes = {
    # "random_spheres": create_random_spheres_scene,
    "material_showcase": create_material_showcase_scene,
}
scenes.update(
    {f"cornell_box_depth_{max_depth}": lambda d=max_depth: create_cornell_box_scene(d) 
     for max_depth in [2, 3, 4, 7, 12]}
)

# Render all scenes
for scene_name, scene_func in scenes.items():
    print(f"Rendering {scene_name}...")
    world, camera = scene_func()
    image = camera.render(world)
    image.save(f"image_{scene_name}.png")
