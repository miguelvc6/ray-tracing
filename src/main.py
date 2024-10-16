import torch as t
from jaxtyping import Float
from camera import Camera
from hittable import HitRecord, Hittable, HittableList  # noqa: F401
from sphere import Sphere

sphere_center: Float[t.Tensor, "3"] = t.tensor([0.0, 0.0, -1.0])
sphere_radius: float = 0.5

sphere: Sphere = Sphere(center=sphere_center, radius=sphere_radius)
ground: Sphere = Sphere(center=t.tensor([0.0, -100.5, -1.0]), radius=100)
world: HittableList = HittableList([sphere, ground])

camera: Camera = Camera(image_width=400, samples_per_pixel=10)
image = camera.render(world)
image.show()
