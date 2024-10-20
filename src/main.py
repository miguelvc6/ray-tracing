import torch as t

from camera import Camera
from hittable import HitRecord, Hittable, HittableList  # noqa: F401
from materials import Dielectric, Lambertian, Metal
from sphere import Sphere

device = t.device("cuda" if t.cuda.is_available() else "cpu")

# Initialize the world with spheres
material_ground = Lambertian(t.Tensor([0.8, 0.8, 0.0]))
material_center = Lambertian(t.Tensor([0.1, 0.2, 0.5]))
material_left = Dielectric(1.50)
material_bubble = Dielectric(1.00 / 1.50)
material_right = Metal(t.Tensor([0.8, 0.6, 0.2]), fuzz=1.0)

world: HittableList = HittableList(
    [
        Sphere(t.tensor([0.0, -100.5, -1.0]), 100.0, material_ground),
        Sphere(t.tensor([0.0, 0.0, -1.2]), 0.5, material_center),
        Sphere(t.tensor([-1.0, 0.0, -1.0]), 0.5, material_left),
        Sphere(t.tensor([-1.0, 0.0, -1.0]), 0.4, material_bubble),
        Sphere(t.tensor([1.0, 0.0, -1.0]), 0.5, material_right),
    ]
)

# Initialize the camera
camera: Camera = Camera(
    image_width=720,
    samples_per_pixel=100,
    aspect_ratio=16.0 / 9.0,
    max_depth=50,
    vfov=90,
    look_from=t.Tensor([-2.0, 2.0, 1.0]),
    look_at=t.Tensor([0.0, 0.0, -1.0]),
    vup=t.Tensor([0.0, 1.0, 0.0]),
    defocus_angle=0.6,
    focus_dist=10.0,
)

# Render the image
image = camera.render(world)
image.save("image.png")
image.show()
