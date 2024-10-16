import torch as t

from camera import Camera
from hittable import HitRecord, Hittable, HittableList  # noqa: F401
from sphere import Sphere

sphere_center = t.tensor([0.0, 0.0, -1.0]) 
sphere_radius = 0.5

sphere = Sphere(center=sphere_center, radius=sphere_radius)
ground = Sphere(center=t.tensor([0.0, -100.5, -1.0]), radius=100)
world = HittableList([sphere, ground])

camera = Camera(image_width=400)
img = camera.render(world)
img.show()