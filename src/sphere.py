import torch as t
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int, jaxtyped
from typeguard import typechecked as typechecker

from config import device
from hittable import HitRecord, Hittable
from materials import Material


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
    def __init__(
        self,
        centers: Float[t.Tensor, "* 3"],
        radii: Float[t.Tensor, "*"],
        material_types: Int[t.Tensor, "*"],
        albedos: Float[t.Tensor, "* 3"],
        fuzzes: Float[t.Tensor, "*"],
        refractive_indices: Float[t.Tensor, "*"],
    ):
        self.centers: Float[t.Tensor, "* 3"] = centers
        self.radii: Float[t.Tensor, "*"] = radii
        self.material_types: Int[t.Tensor, "*"] = material_types
        self.albedos: Float[t.Tensor, "* 3"] = albedos
        self.fuzzes: Float[t.Tensor, "*"] = fuzzes
        self.refractive_indices: Float[t.Tensor, "*"] = refractive_indices

    def hit(self, pixel_rays: Float[t.Tensor, "N 3 2"], t_min: float, t_max: float) -> HitRecord:
        N: int = pixel_rays.shape[0]
        M: int = self.centers.shape[0]
        rays_origin: Float[t.Tensor, "N 3"] = pixel_rays[:, :, 0]
        rays_direction: Float[t.Tensor, "N 3"] = pixel_rays[:, :, 1]

        rays_origin: Float[t.Tensor, "N M 3"] = rays_origin.unsqueeze(1).expand(-1, M, -1)
        rays_direction: Float[t.Tensor, "N M 3"] = rays_direction.unsqueeze(1).expand(-1, M, -1)
        centers: Float[t.Tensor, "N M 3"] = self.centers.unsqueeze(0).expand(N, -1, -1)
        radii: Float[t.Tensor, "N M"] = self.radii.unsqueeze(0).expand(N, -1)

        oc: Float[t.Tensor, "N M 3"] = rays_origin - centers

        a: Float[t.Tensor, "N M"] = (rays_direction**2).sum(dim=2)
        b: Float[t.Tensor, "N M"] = 2.0 * (rays_direction * oc).sum(dim=2)
        c: Float[t.Tensor, "N M"] = (oc**2).sum(dim=2) - radii**2

        discriminant: Float[t.Tensor, "N M"] = b**2 - 4 * a * c
        valid_discriminant: Bool[t.Tensor, "N M"] = discriminant >= 0

        sqrt_discriminant: Float[t.Tensor, "N M"] = t.zeros_like(discriminant)
        sqrt_discriminant[valid_discriminant] = t.sqrt(discriminant[valid_discriminant])

        denom: Float[t.Tensor, "N M"] = 2.0 * a
        t0: Float[t.Tensor, "N M"] = t.full_like(discriminant, float("inf"))
        t1: Float[t.Tensor, "N M"] = t.full_like(discriminant, float("inf"))

        t0[valid_discriminant] = (-b[valid_discriminant] - sqrt_discriminant[valid_discriminant]) / denom[
            valid_discriminant
        ]
        t1[valid_discriminant] = (-b[valid_discriminant] + sqrt_discriminant[valid_discriminant]) / denom[
            valid_discriminant
        ]

        t0_valid: Bool[t.Tensor, "N M"] = (t0 > t_min) & (t0 < t_max)
        t1_valid: Bool[t.Tensor, "N M"] = (t1 > t_min) & (t1 < t_max)

        t_hit: Float[t.Tensor, "N M"] = t.full_like(discriminant, float("inf"))
        t_hit[t0_valid] = t0[t0_valid]
        t_hit[t1_valid & (t1 < t_hit)] = t1[t1_valid & (t1 < t_hit)]

        sphere_hit: Bool[t.Tensor, "N M"] = valid_discriminant & (t_hit < float("inf"))

        t_hit_min: Float[t.Tensor, "N"] = t.min(t_hit, dim=1)[0]
        sphere_indices: Int[t.Tensor, "N"] = t.min(t_hit, dim=1)[1]
        sphere_hit_any: Bool[t.Tensor, "N"] = sphere_hit.any(dim=1)
        t_hit_min[~sphere_hit_any] = float("inf")

        record: HitRecord = HitRecord.empty((N,))
        record.hit = sphere_hit_any
        record.t[sphere_hit_any] = t_hit_min[sphere_hit_any]
        rays_direction: Float[t.Tensor, "N 3"] = rays_direction[:, 0, :]
        rays_origin: Float[t.Tensor, "N 3"] = rays_origin[:, 0, :]
        hit_points: Float[t.Tensor, "N 3"] = rays_origin + rays_direction * t_hit_min.unsqueeze(1)
        centers_hit: Float[t.Tensor, "N 3"] = self.centers[sphere_indices]
        normal_vectors: Float[t.Tensor, "N 3"] = F.normalize(hit_points - centers_hit, dim=1)
        record.point[sphere_hit_any] = hit_points[sphere_hit_any]
        record.normal[sphere_hit_any] = normal_vectors[sphere_hit_any]
        record.set_face_normal(rays_direction, record.normal)

        record.material_type[sphere_hit_any] = self.material_types[sphere_indices[sphere_hit_any]]
        record.albedo[sphere_hit_any] = self.albedos[sphere_indices[sphere_hit_any]]
        record.fuzz[sphere_hit_any] = self.fuzzes[sphere_indices[sphere_hit_any]]
        record.refractive_index[sphere_hit_any] = self.refractive_indices[sphere_indices[sphere_hit_any]]

        return record
