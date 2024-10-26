import math
import random
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import List

import numpy as np
import torch as t
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int, jaxtyped
from PIL import Image
from tqdm import tqdm
from typeguard import typechecked as typechecker

device = t.device("cuda" if t.cuda.is_available() else "cpu")



# FILE 1/6: utils.py



@jaxtyped(typechecker=typechecker)
def tensor_to_image(tensor: Float[t.Tensor, "h w c"] | Int[t.Tensor, "h w c"]) -> Image.Image:
    tensor = tensor.sqrt()  # gamma correction
    tensor = tensor.multiply(255).clamp(0, 255)
    array = tensor.cpu().numpy().astype(np.uint8)
    array = array[::-1, :, :]
    image = Image.fromarray(array, mode="RGB")
    return image


@jaxtyped(typechecker=typechecker)
def degrees_to_radians(degrees: float) -> float:
    return degrees * np.pi / 180.0


@jaxtyped(typechecker=typechecker)
def random_unit_vector(shape: tuple[int, ...]) -> Float[t.Tensor, "... 3"]:
    vec = t.randn(*shape, device=device)
    vec = F.normalize(vec, dim=-1)
    return vec


@jaxtyped(typechecker=typechecker)
def random_on_hemisphere(normal: Float[t.Tensor, "... 3"]) -> Float[t.Tensor, "... 3"]:
    vec = random_unit_vector(normal.shape)
    dot_product = t.sum(vec * normal, dim=-1, keepdim=True)
    return t.where(dot_product > 0, vec, -vec)


@jaxtyped(typechecker=typechecker)
def background_color_gradient(sample: int, h: int, w: int) -> Float[t.Tensor, "sample h w 3"]:
    white: Float[t.Tensor, "3"] = t.tensor([1.0, 1.0, 1.0], device=device)
    light_blue: Float[t.Tensor, "3"] = t.tensor([0.5, 0.7, 1.0], device=device)
    a: Float[t.Tensor, "h 1"] = t.linspace(0, 1, h, device=device).unsqueeze(1)
    background_colors_single: Float[t.Tensor, "h 3"] = a * light_blue + (1.0 - a) * white
    background_colors: Float[t.Tensor, "sample h w 3"] = (
        background_colors_single.unsqueeze(0).unsqueeze(2).expand(sample, h, w, 3) * 255
    )
    return background_colors


@jaxtyped(typechecker=typechecker)
def random_in_unit_disk(shape: tuple[int, ...]) -> Float[t.Tensor, "... 2"]:
    r: Float[t.Tensor, "..."] = t.sqrt(t.rand(*shape, device=device))
    theta: Float[t.Tensor, "..."] = t.rand(*shape, device=device) * 2 * np.pi
    x: Float[t.Tensor, "..."] = r * t.cos(theta)
    y: Float[t.Tensor, "..."] = r * t.sin(theta)
    return t.stack([x, y], dim=-1)


# FILE 2/6: materials.py





class MaterialType(IntEnum):
    Lambertian = 0
    Metal = 1
    Dielectric = 2


@jaxtyped(typechecker=typechecker)
def reflect(v: Float[t.Tensor, "N 3"], n: Float[t.Tensor, "N 3"]) -> Float[t.Tensor, "N 3"]:
    # Reflects vector v around normal n
    return v - 2 * (v * n).sum(dim=1, keepdim=True) * n


@jaxtyped(typechecker=typechecker)
def refract(
    uv: Float[t.Tensor, "N 3"], n: Float[t.Tensor, "N 3"], etai_over_etat: Float[t.Tensor, "N 1"]
) -> Float[t.Tensor, "N 3"]:
    one = t.tensor(1.0, device=uv.device)
    cos_theta = t.minimum((-uv * n).sum(dim=1, keepdim=True), one)
    r_out_perp = etai_over_etat * (uv + cos_theta * n)
    r_out_parallel = -t.sqrt(t.abs(one - (r_out_perp**2).sum(dim=1, keepdim=True))) * n
    return r_out_perp + r_out_parallel


@jaxtyped(typechecker=typechecker)
def reflectance(cosine: Float[t.Tensor, "N 1"], ref_idx: Float[t.Tensor, "N 1"]) -> Float[t.Tensor, "N 1"]:
    one = t.tensor(1.0, device=ref_idx.device)
    r0 = ((one - ref_idx) / (one + ref_idx)) ** 2
    return r0 + (one - r0) * (one - cosine) ** 5


@jaxtyped(typechecker=typechecker)
class Material(ABC):
    @jaxtyped(typechecker=typechecker)
    def __init__(self):
        pass

    @abstractmethod
    @jaxtyped(typechecker=typechecker)
    def scatter(
        self,
        r_in: Float[t.Tensor, "* 3 2"],
        hit_record: "HitRecord",
    ) -> tuple[
        Bool[t.Tensor, "*"],
        Float[t.Tensor, "* 3"],
        Float[t.Tensor, "* 3 2"],
    ]:
        pass


@jaxtyped(typechecker=typechecker)
class Lambertian(Material):
    def __init__(self, albedo: Float[t.Tensor, "3"]):
        self.albedo = albedo.to(device)

    @jaxtyped(typechecker=typechecker)
    def scatter(
        self,
        r_in: Float[t.Tensor, "N 3 2"],
        hit_record: "HitRecord",
    ) -> tuple[
        Bool[t.Tensor, "*"],
        Float[t.Tensor, "* 3"],
        Float[t.Tensor, "* 3 2"],
    ]:
        N = r_in.shape[0]
        normals = hit_record.normal
        points = hit_record.point

        # Generate scatter direction
        scatter_direction = normals + random_unit_vector((N, 3)).to(device)

        # Handle degenerate scatter direction
        zero_mask = scatter_direction.norm(dim=1) < 1e-8
        scatter_direction[zero_mask] = normals[zero_mask]

        # Normalize scatter direction
        scatter_direction = F.normalize(scatter_direction, dim=-1)

        # Create new rays for recursion
        new_origin = points
        new_direction = scatter_direction
        new_rays = t.stack([new_origin, new_direction], dim=-1)

        # Attenuation is the albedo
        attenuation = hit_record.albedo

        scatter_mask = t.ones(N, dtype=t.bool, device=device)

        return scatter_mask, attenuation, new_rays


@jaxtyped(typechecker=typechecker)
class Metal(Material):
    def __init__(self, albedo: Float[t.Tensor, "3"], fuzz: float = 0.3):
        self.albedo = albedo.to(device)
        self.fuzz = max(0.0, min(fuzz, 1.0))

    @jaxtyped(typechecker=typechecker)
    def scatter(
        self,
        r_in: Float[t.Tensor, "N 3 2"],
        hit_record: "HitRecord",
    ) -> tuple[
        Bool[t.Tensor, "*"],
        Float[t.Tensor, "N 3"],
        Float[t.Tensor, "N 3 2"],
    ]:
        N = r_in.shape[0]
        normals = hit_record.normal  # Shape: [N, 3]
        points = hit_record.point  # Shape: [N, 3]

        # Incoming ray directions
        in_directions = r_in[:, :, 1]  # Shape: [N, 3]
        in_directions = F.normalize(in_directions, dim=-1)

        # Generate reflected directions
        reflected_direction = reflect(in_directions, normals)

        reflected_direction = reflected_direction + self.fuzz * random_unit_vector((N, 3)).to(device)
        reflected_direction = F.normalize(reflected_direction, dim=-1)

        # Check if reflected ray is above the surface
        dot_product = t.sum(reflected_direction * normals, dim=1)  # Shape: [N]
        scatter_mask = dot_product > 0  # Shape: [N], dtype: bool

        # Create new rays for recursion
        new_origin = points  # Shape: [N, 3]
        new_direction = reflected_direction  # Shape: [N, 3]
        new_rays = t.stack([new_origin, new_direction], dim=-1)  # Shape: [N, 3, 2]

        # Attenuation is the albedo
        attenuation = hit_record.albedo

        return scatter_mask, attenuation, new_rays


@jaxtyped(typechecker=typechecker)
class Dielectric(Material):
    def __init__(self, refraction_index: float):
        self.refraction_index = refraction_index

    @jaxtyped(typechecker=typechecker)
    def scatter(
        self,
        r_in: Float[t.Tensor, "N 3 2"],
        hit_record: "HitRecord",
    ) -> tuple[
        Bool[t.Tensor, "*"],
        Float[t.Tensor, "N 3"],
        Float[t.Tensor, "N 3 2"],
    ]:
        N = r_in.shape[0]
        normals = hit_record.normal  # Shape: [N, 3]
        points = hit_record.point  # Shape: [N, 3]
        front_face = hit_record.front_face  # Shape: [N], dtype: bool
        unit_direction = F.normalize(r_in[:, :, 1], dim=1)  # Shape: [N, 3]

        # Attenuation is always (1, 1, 1) for dielectric materials
        attenuation = t.ones(N, 3, device=device)  # Shape: [N, 3]

        one = t.tensor(1.0, device=device)
        refraction_ratio = t.where(
            front_face.unsqueeze(1),
            t.full((N, 1), 1.0 / self.refraction_index, device=device),
            t.full((N, 1), self.refraction_index, device=device),
        )

        cos_theta = t.minimum((-unit_direction * normals).sum(dim=1, keepdim=True), one)
        sin_theta = t.sqrt(one - cos_theta**2)

        cannot_refract = (refraction_ratio * sin_theta) > one

        # Generate random numbers to decide between reflection and refraction
        reflect_prob = reflectance(cos_theta, refraction_ratio)
        random_numbers = t.rand(N, 1, device=device)
        should_reflect = cannot_refract | (reflect_prob > random_numbers)

        # Compute reflected and refracted directions
        reflected_direction = reflect(unit_direction, normals)
        refracted_direction = refract(unit_direction, normals, refraction_ratio)
        direction = t.where(should_reflect.expand(-1, 3), reflected_direction, refracted_direction)
        new_rays = t.stack([points, direction], dim=-1)

        # Scatter mask is always True for dielectric materials
        scatter_mask = t.ones(N, dtype=t.bool, device=device)

        return scatter_mask, attenuation, new_rays


# FILE 3/6: hittable.py




@jaxtyped(typechecker=typechecker)
class HitRecord:
    @jaxtyped(typechecker=typechecker)
    def __init__(
        self, hit, point, normal, t, front_face=None, material_type=None, albedo=None, fuzz=None, refractive_index=None
    ):
        self.hit = hit
        self.point = point
        self.normal = normal
        self.t = t
        self.front_face = front_face
        self.material_type = material_type
        self.albedo = albedo
        self.fuzz = fuzz
        self.refractive_index = refractive_index

    @jaxtyped(typechecker=typechecker)
    def set_face_normal(
        self,
        ray_direction: Float[t.Tensor, "... 3"],
        outward_normal: Float[t.Tensor, "... 3"],
    ) -> None:
        """Determines whether the hit is from the outside or inside."""
        self.front_face: Bool[t.Tensor, "..."] = (ray_direction * outward_normal).sum(dim=-1) < 0
        self.normal: Float[t.Tensor, "... 3"] = t.where(self.front_face.unsqueeze(-1), outward_normal, -outward_normal)

    @staticmethod
    @jaxtyped(typechecker=typechecker)
    def empty(shape):
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
        hit = t.full(shape, False, dtype=t.bool, device=device)
        point = t.zeros((*shape, 3), dtype=t.float32, device=device)
        normal = t.zeros((*shape, 3), dtype=t.float32, device=device)
        t_values = t.full(shape, float("inf"), dtype=t.float32, device=device)
        front_face = t.full(shape, False, dtype=t.bool, device=device)
        material_type = t.full(shape, -1, dtype=t.int32, device=device)
        albedo = t.zeros((*shape, 3), dtype=t.float32, device=device)
        fuzz = t.zeros(shape, dtype=t.float32, device=device)
        refractive_index = t.zeros(shape, dtype=t.float32, device=device)
        return HitRecord(hit, point, normal, t_values, front_face, material_type, albedo, fuzz, refractive_index)


@jaxtyped(typechecker=typechecker)
class Hittable(ABC):
    """Abstract class for hittable objects."""

    @abstractmethod
    @jaxtyped(typechecker=typechecker)
    def hit(
        self,
        pixel_rays: Float[t.Tensor, "N 3 2"],
        t_min: float,
        t_max: float,
    ) -> HitRecord:
        pass


@jaxtyped(typechecker=typechecker)
class HittableList(Hittable):
    """List of hittable objects."""

    def __init__(self, objects: List[Hittable] = []):
        self.objects: List[Hittable] = objects

    def add(self, object: Hittable) -> None:
        self.objects.append(object)

    def hit(
        self,
        pixel_rays: Float[t.Tensor, "N 3 2"],
        t_min: float,
        t_max: float,
    ) -> HitRecord:

        N: int = pixel_rays.shape[0]
        record: HitRecord = HitRecord.empty((N,))
        closest_so_far: Float[t.Tensor, "N"] = t.full((N,), t_max, device=device)

        for obj in self.objects:
            obj_record: HitRecord = obj.hit(pixel_rays, t_min, t_max)
            closer_mask: Bool[t.Tensor, "N"] = obj_record.hit & (obj_record.t < closest_so_far)
            closest_so_far = t.where(closer_mask, obj_record.t, closest_so_far)

            record.hit = record.hit | obj_record.hit
            record.point = t.where(closer_mask.unsqueeze(-1), obj_record.point, record.point)
            record.normal = t.where(closer_mask.unsqueeze(-1), obj_record.normal, record.normal)
            record.t = t.where(closer_mask, obj_record.t, record.t)
            record.front_face = t.where(closer_mask, obj_record.front_face, record.front_face)

            # Update materials
            indices = closer_mask.nonzero(as_tuple=False).squeeze(-1)
            for idx in indices.tolist():
                record.material[idx] = obj_record.material[idx]

        return record


# FILE 4/6: sphere.py



@jaxtyped(typechecker=typechecker)
class Sphere(Hittable):
    def __init__(self, center: Float[t.Tensor, "3"], radius: float, material: Material):
        self.center: Float[t.Tensor, "3"] = center.to(device)
        self.radius: float = max(radius, 0.0)
        self.material: Material = material

    def hit(
        self,
        pixel_rays: Float[t.Tensor, "N 3 2"],
        t_min: float,
        t_max: float,
    ) -> HitRecord:
        N: int = pixel_rays.shape[0]
        record: HitRecord = HitRecord.empty((N,))

        origin: Float[t.Tensor, "N 3"] = pixel_rays[:, :, 0]
        pixel_directions: Float[t.Tensor, "N 3"] = pixel_rays[:, :, 1]

        oc: Float[t.Tensor, "N 3"] = origin - self.center

        # Solve quadratic equation
        a: Float[t.Tensor, "N"] = (pixel_directions**2).sum(dim=1)
        b: Float[t.Tensor, "N"] = 2.0 * (pixel_directions * oc).sum(dim=1)
        c: Float[t.Tensor, "N"] = (oc**2).sum(dim=1) - self.radius**2

        discriminant: Float[t.Tensor, "N"] = b**2 - 4 * a * c
        sphere_hit: Bool[t.Tensor, "N"] = discriminant >= 0

        t_hit: Float[t.Tensor, "N"] = t.full((N,), float("inf"), device=device)
        sqrt_discriminant: Float[t.Tensor, "N"] = t.zeros(N, device=device)
        sqrt_discriminant[sphere_hit] = t.sqrt(discriminant[sphere_hit])

        # Compute roots
        t0: Float[t.Tensor, "N"] = t.zeros(N, device=device)
        t1: Float[t.Tensor, "N"] = t.zeros(N, device=device)
        denom: Float[t.Tensor, "N"] = 2.0 * a
        t0[sphere_hit] = (-b[sphere_hit] - sqrt_discriminant[sphere_hit]) / denom[sphere_hit]
        t1[sphere_hit] = (-b[sphere_hit] + sqrt_discriminant[sphere_hit]) / denom[sphere_hit]

        t0_valid: Bool[t.Tensor, "N"] = (t0 > t_min) & (t0 < t_max)
        t1_valid: Bool[t.Tensor, "N"] = (t1 > t_min) & (t1 < t_max)

        t_hit = t.where((t0_valid) & (t0 < t_hit), t0, t_hit)
        t_hit = t.where((t1_valid) & (t1 < t_hit), t1, t_hit)

        sphere_hit = sphere_hit & (t_hit < float("inf"))

        # Compute hit points and normals where sphere_hit is True
        hit_points: Float[t.Tensor, "N 3"] = origin + pixel_directions * t_hit.unsqueeze(-1)
        normal_vectors: Float[t.Tensor, "N 3"] = F.normalize(hit_points - self.center, dim=1)

        # Update the record
        record.hit = sphere_hit
        record.t[sphere_hit] = t_hit[sphere_hit]
        record.point[sphere_hit] = hit_points[sphere_hit]
        record.normal[sphere_hit] = normal_vectors[sphere_hit]
        record.set_face_normal(pixel_directions, record.normal)

        # Set material for hits
        indices = sphere_hit.nonzero(as_tuple=False).squeeze(-1)
        for idx in indices:
            record.material[idx] = self.material
        return record


class SphereList(Hittable):
    def __init__(self, centers, radii, material_types, albedos, fuzzes, refractive_indices):
        self.centers = centers
        self.radii = radii
        self.material_types = material_types
        self.albedos = albedos
        self.fuzzes = fuzzes
        self.refractive_indices = refractive_indices

    def hit(self, pixel_rays, t_min, t_max):
        N = pixel_rays.shape[0]
        M = self.centers.shape[0]
        rays_origin = pixel_rays[:, :, 0]  # [N, 3]
        rays_direction = pixel_rays[:, :, 1]  # [N, 3]

        # Expand to match the number of spheres
        rays_origin = rays_origin.unsqueeze(1).expand(-1, M, -1)  # [N, M, 3]
        rays_direction = rays_direction.unsqueeze(1).expand(-1, M, -1)  # [N, M, 3]
        centers = self.centers.unsqueeze(0).expand(N, -1, -1)  # [N, M, 3]
        radii = self.radii.unsqueeze(0).expand(N, -1)  # [N, M]

        oc = rays_origin - centers  # [N, M, 3]

        a = (rays_direction**2).sum(dim=2)  # [N, M]
        b = 2.0 * (rays_direction * oc).sum(dim=2)  # [N, M]
        c = (oc**2).sum(dim=2) - radii**2  # [N, M]

        discriminant = b**2 - 4 * a * c  # [N, M]
        valid_discriminant = discriminant >= 0  # [N, M]

        sqrt_discriminant = t.zeros_like(discriminant)
        sqrt_discriminant[valid_discriminant] = t.sqrt(discriminant[valid_discriminant])

        denom = 2.0 * a  # [N, M]
        t0 = t.full_like(discriminant, float("inf"))
        t1 = t.full_like(discriminant, float("inf"))

        t0[valid_discriminant] = (-b[valid_discriminant] - sqrt_discriminant[valid_discriminant]) / denom[
            valid_discriminant
        ]
        t1[valid_discriminant] = (-b[valid_discriminant] + sqrt_discriminant[valid_discriminant]) / denom[
            valid_discriminant
        ]

        t0_valid = (t0 > t_min) & (t0 < t_max)
        t1_valid = (t1 > t_min) & (t1 < t_max)

        t_hit = t.full_like(discriminant, float("inf"))
        t_hit[t0_valid] = t0[t0_valid]
        t_hit[t1_valid & (t1 < t_hit)] = t1[t1_valid & (t1 < t_hit)]

        sphere_hit = valid_discriminant & (t_hit < float("inf"))

        # Find the closest hit for each ray
        t_hit_min, sphere_indices = t.min(t_hit, dim=1)  # [N]
        sphere_hit_any = sphere_hit.any(dim=1)  # [N]
        t_hit_min[~sphere_hit_any] = float("inf")

        # Prepare the hit record
        record = HitRecord.empty((N,))
        record.hit = sphere_hit_any
        record.t[sphere_hit_any] = t_hit_min[sphere_hit_any]
        rays_direction = rays_direction[:, 0, :]  # [N, 3]
        rays_origin = rays_origin[:, 0, :]  # [N, 3]
        hit_points = rays_origin + rays_direction * t_hit_min.unsqueeze(1)  # [N, 3]
        centers_hit = self.centers[sphere_indices]  # [N, 3]
        normal_vectors = F.normalize(hit_points - centers_hit, dim=1)  # [N, 3]
        record.point[sphere_hit_any] = hit_points[sphere_hit_any]
        record.normal[sphere_hit_any] = normal_vectors[sphere_hit_any]
        record.set_face_normal(rays_direction, record.normal)

        # Set per-ray material data
        record.material_type[sphere_hit_any] = self.material_types[sphere_indices[sphere_hit_any]]
        record.albedo[sphere_hit_any] = self.albedos[sphere_indices[sphere_hit_any]]
        record.fuzz[sphere_hit_any] = self.fuzzes[sphere_indices[sphere_hit_any]]
        record.refractive_index[sphere_hit_any] = self.refractive_indices[sphere_indices[sphere_hit_any]]

        return record


# FILE 5/6: camera.py




@jaxtyped(typechecker=typechecker)
class Camera:
    def __init__(
        self,
        look_from: Float[t.Tensor, "3"] = t.tensor([0.0, 0.0, 0.0], device=device),
        look_at: Float[t.Tensor, "3"] = t.tensor([0.0, 0.0, -1.0], device=device),
        vup: Float[t.Tensor, "3"] = t.tensor([0.0, 1.0, 0.0], device=device),
        aspect_ratio: float = 16.0 / 9.0,
        image_width: int = 400,
        samples_per_pixel: int = 10,
        max_depth: int = 50,
        vfov: float = 90.0,  # vertical field of view angle
        defocus_angle: float = 0,
        focus_dist: float = 10.0,
    ):
        self.look_from: Float[t.Tensor, "3"] = look_from
        self.look_at: Float[t.Tensor, "3"] = look_at
        self.vup: Float[t.Tensor, "3"] = vup

        self.samples_per_pixel: int = samples_per_pixel
        self.max_depth: int = max_depth

        self.defocus_angle: float = defocus_angle
        self.focus_dist: float = focus_dist

        self.aspect_ratio: float = aspect_ratio
        self.image_width: int = image_width
        self.image_height: int = max(1, int(image_width / aspect_ratio))
        h, w = self.image_height, self.image_width

        # Compute viewport dimensions
        theta: float = degrees_to_radians(vfov)
        h_viewport: float = math.tan(theta / 2)
        self.viewport_height: float = 2.0 * h_viewport * focus_dist
        self.viewport_width: float = self.viewport_height * self.aspect_ratio

        # Calculate camera basis vectors
        self.w: Float[t.Tensor, "3"] = F.normalize(self.look_from - self.look_at, dim=-1)
        self.u: Float[t.Tensor, "3"] = F.normalize(t.cross(self.vup, self.w, dim=-1), dim=-1)
        self.v: Float[t.Tensor, "3"] = t.cross(self.w, self.u, dim=-1)

        # Calculate viewport dimensions and vectors
        viewport_u: Float[t.Tensor, "3"] = self.viewport_width * self.u
        viewport_v: Float[t.Tensor, "3"] = self.viewport_height * self.v

        # Calculate the lower-left corner of the viewport
        self.viewport_lower_left: Float[t.Tensor, "3"] = (
            self.look_from - viewport_u / 2 - viewport_v / 2 - self.w * focus_dist
        )

        # Calculate the camera defocus disk basis vectors
        defocus_radius: float = focus_dist * math.tan(degrees_to_radians(defocus_angle / 2))
        self.defocus_disk_u: Float[t.Tensor, "3"] = self.u * defocus_radius
        self.defocus_disk_v: Float[t.Tensor, "3"] = self.v * defocus_radius

        # Store pixel delta vectors
        self.pixel_delta_u: Float[t.Tensor, "3"] = viewport_u / (w - 1)
        self.pixel_delta_v: Float[t.Tensor, "3"] = viewport_v / (h - 1)

        self.viewport_lower_left = self.viewport_lower_left.to(device)
        self.pixel_delta_u = self.pixel_delta_u.to(device)
        self.pixel_delta_v = self.pixel_delta_v.to(device)

    @jaxtyped(typechecker=typechecker)
    def defocus_disk_sample(self, sample: int, h: int, w: int) -> Float[t.Tensor, "sample h w 3"]:
        p: Float[t.Tensor, "sample h w 2"] = random_in_unit_disk((sample, h, w))
        offset = p[..., 0].unsqueeze(-1) * self.defocus_disk_u.view(1, 1, 1, 3) + p[..., 1].unsqueeze(
            -1
        ) * self.defocus_disk_v.view(1, 1, 1, 3)
        return self.look_from.view(1, 1, 1, 3) + offset

    @jaxtyped(typechecker=typechecker)
    def ray_color(
        self,
        pixel_rays: Float[t.Tensor, "N 3 2"],
        world: Hittable,
    ) -> Float[t.Tensor, "N 3"]:
        N = pixel_rays.shape[0]
        colors = t.zeros((N, 3), device=device)
        attenuation = t.ones((N, 3), device=device)
        rays = pixel_rays
        active_mask = t.ones(N, dtype=t.bool, device=device)

        for depth in tqdm(range(self.max_depth), total=self.max_depth):
            if not active_mask.any():
                break

            # Perform hit test
            hit_record = world.hit(rays, 0.001, float("inf"))

            # Handle rays that did not hit anything
            no_hit_mask = (~hit_record.hit) & active_mask
            if no_hit_mask.any():
                ray_dirs = F.normalize(rays[no_hit_mask, :, 1], dim=-1)
                t_param = 0.5 * (ray_dirs[:, 1] + 1.0)
                background_colors = (1.0 - t_param).unsqueeze(-1) * t.tensor([1.0, 1.0, 1.0], device=device)
                background_colors += t_param.unsqueeze(-1) * t.tensor([0.5, 0.7, 1.0], device=device)
                colors[no_hit_mask] += attenuation[no_hit_mask] * background_colors
                active_mask[no_hit_mask] = False

            # Handle rays that hit an object
            hit_mask = hit_record.hit & active_mask
            if hit_mask.any():
                hit_indices = hit_mask.nonzero(as_tuple=False).squeeze(-1)
                material_types_hit = hit_record.material_type[hit_indices]

                # Group indices by material
                for material_type in [MaterialType.Lambertian, MaterialType.Metal, MaterialType.Dielectric]:
                    material_mask = material_types_hit == material_type

                    if material_mask.any():
                        indices = hit_indices[material_mask]
                        ray_in = rays[indices]
                        sub_hit_record = HitRecord(
                            hit=hit_record.hit[indices],
                            point=hit_record.point[indices],
                            normal=hit_record.normal[indices],
                            t=hit_record.t[indices],
                            front_face=hit_record.front_face[indices],
                            material_type=hit_record.material_type[indices],
                            albedo=hit_record.albedo[indices],
                            fuzz=hit_record.fuzz[indices],
                            refractive_index=hit_record.refractive_index[indices],
                        )

                    # Process scattering for each material
                    if material_type == MaterialType.Lambertian:
                        scatter_mask, mat_attenuation, scattered_rays = Lambertian.scatter_material(
                            ray_in, sub_hit_record
                        )
                    elif material_type == MaterialType.Metal:
                        scatter_mask, mat_attenuation, scattered_rays = Metal.scatter_material(ray_in, sub_hit_record)
                    elif material_type == MaterialType.Dielectric:
                        scatter_mask, mat_attenuation, scattered_rays = Dielectric.scatter_material(
                            ray_in, sub_hit_record
                        )
                    attenuation[indices] *= mat_attenuation
                    rays[indices] = scattered_rays
                    terminated = ~scatter_mask
                    if terminated.any():
                        term_indices = indices[terminated.nonzero(as_tuple=False).squeeze(-1)]
                        colors[term_indices] += attenuation[term_indices] * t.zeros(
                            (term_indices.numel(), 3), device=device
                        )
                        active_mask[term_indices] = False
            else:
                break
        # Any remaining active rays contribute background color
        if active_mask.any():
            ray_dirs = F.normalize(rays[active_mask, :, 1], dim=-1)
            t_param = 0.5 * (ray_dirs[:, 1] + 1.0)
            background_colors = (1.0 - t_param).unsqueeze(-1) * t.tensor([1.0, 1.0, 1.0], device=device)
            background_colors += t_param.unsqueeze(-1) * t.tensor([0.5, 0.7, 1.0], device=device)
            colors[active_mask] += attenuation[active_mask] * background_colors

        return colors

    @jaxtyped(typechecker=typechecker)
    def render(self, world: Hittable) -> Image.Image:
        sample: int = self.samples_per_pixel
        h, w = self.image_height, self.image_width

        # Prepare grid indices for pixels
        j_indices = t.arange(h, device=device).view(1, h, 1, 1)
        i_indices = t.arange(w, device=device).view(1, 1, w, 1)

        # Generate random offsets for antialiasing
        noise_u: Float[t.Tensor, "sample h w 1"] = t.rand((sample, h, w, 1), device=device)
        noise_v: Float[t.Tensor, "sample h w 1"] = t.rand((sample, h, w, 1), device=device)

        # Compute pixel positions on the viewport
        sampled_pixels: Float[t.Tensor, "sample h w 3"] = (
            self.viewport_lower_left.view(1, 1, 1, 3)
            + (i_indices + noise_u) * self.pixel_delta_u.view(1, 1, 1, 3)
            + (j_indices + noise_v) * self.pixel_delta_v.view(1, 1, 1, 3)
        )

        # Compute direction vectors
        directions: Float[t.Tensor, "sample h w 3"] = F.normalize(
            sampled_pixels - self.look_from.to(device).view(1, 1, 1, 3), dim=-1
        )

        # Build rays
        origin: Float[t.Tensor, "sample h w 3"] = self.look_from.to(device).view(1, 1, 1, 3).expand(sample, h, w, 3)
        if self.defocus_angle <= 0:
            ray_origin = origin
        else:
            ray_origin = self.defocus_disk_sample(sample, h, w)

        pixel_rays: Float[t.Tensor, "sample h w 3 2"] = t.stack([ray_origin, directions], dim=-1)

        # Flatten rays for processing
        N = sample * h * w
        pixel_rays_flat: Float[t.Tensor, "N 3 2"] = pixel_rays.view(N, 3, 2)

        # Prepare an empty tensor for colors
        colors_flat = t.zeros((N, 3), device=device)

        # Define batch size
        batch_size = 10000  # Adjust this value based on your GPU memory

        # Process rays in batches
        for i in tqdm(range(0, N, batch_size), total=(N + batch_size - 1) // batch_size):
            rays_batch = pixel_rays_flat[i : i + batch_size]
            colors_batch = self.ray_color(rays_batch, world)
            colors_flat[i : i + batch_size] = colors_batch

        colors: Float[t.Tensor, "sample h w 3"] = colors_flat.view(sample, h, w, 3)

        # Average over antialiasing samples
        img: Float[t.Tensor, "h w 3"] = colors.mean(dim=0)

        # Convert to image
        return tensor_to_image(img)


# FILE 6/6: main.py



# Choose device
print(f"Using device: {device}")


def random_double(min_val=0.0, max_val=1.0):
    return min_val + (max_val - min_val) * random.random()


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
    image_width=400,
    samples_per_pixel=10,
    aspect_ratio=16.0 / 9.0,
    max_depth=10,
    vfov=20,
    look_from=t.tensor([13, 2, 3], dtype=t.float32, device=device),
    look_at=t.tensor([0, 0, 0], dtype=t.float32, device=device),
    vup=t.tensor([0, 1, 0], dtype=t.float32, device=device),
    defocus_angle=0.,
    focus_dist=10.0,
)


# Render the image
image = camera.render(world)
image.save("image.png")
image.show()
