import torch as t
import torch.nn.functional as F
from jaxtyping import Bool, Float, jaxtyped
from typeguard import typechecked as typechecker

from hittable import HitRecord, Hittable

@jaxtyped(typechecker=typechecker)
class Sphere(Hittable):
    def __init__(self, center: Float[t.Tensor, "3"], radius: float):
        self.center: Float[t.Tensor, "3"] = center
        self.radius: float = max(radius, 0.0)

    def hit(
        self,
        pixel_rays: Float[t.Tensor, "N 3 2"],
        t_min: float,
        t_max: float,
    ) -> HitRecord:
        N: int = pixel_rays.shape[0]
        record: HitRecord = HitRecord.empty((N,), device=pixel_rays.device)

        origin: Float[t.Tensor, "N 3"] = pixel_rays[:, :, 0]  # Shape: [N, 3]
        pixel_directions: Float[t.Tensor, "N 3"] = pixel_rays[:, :, 1]  # Shape: [N, 3]

        oc: Float[t.Tensor, "N 3"] = origin - self.center  # Broadcasted to [N, 3]

        a: Float[t.Tensor, "N"] = (pixel_directions ** 2).sum(dim=1)
        b: Float[t.Tensor, "N"] = 2.0 * (pixel_directions * oc).sum(dim=1)
        c: Float[t.Tensor, "N"] = (oc ** 2).sum(dim=1) - self.radius ** 2

        discriminant: Float[t.Tensor, "N"] = b ** 2 - 4 * a * c
        sphere_hit: Bool[t.Tensor, "N"] = discriminant >= 0

        t_hit: Float[t.Tensor, "N"] = t.full_like(discriminant, float('inf'))
        sqrt_discriminant: Float[t.Tensor, "N"] = t.zeros_like(discriminant)
        sqrt_discriminant[sphere_hit] = t.sqrt(discriminant[sphere_hit])

        t0: Float[t.Tensor, "N"] = t.zeros_like(discriminant)
        t1: Float[t.Tensor, "N"] = t.zeros_like(discriminant)
        denom: Float[t.Tensor, "N"] = 2.0 * a
        t0[sphere_hit] = (-b[sphere_hit] - sqrt_discriminant[sphere_hit]) / denom[sphere_hit]
        t1[sphere_hit] = (-b[sphere_hit] + sqrt_discriminant[sphere_hit]) / denom[sphere_hit]

        t0_valid: Bool[t.Tensor, "N"] = (t0 > t_min) & (t0 < t_max)
        t1_valid: Bool[t.Tensor, "N"] = (t1 > t_min) & (t1 < t_max)

        t_hit = t.where((t0_valid) & (t0 < t_hit), t0, t_hit)
        t_hit = t.where((t1_valid) & (t1 < t_hit), t1, t_hit)

        sphere_hit = sphere_hit & (t_hit < float('inf'))

        # Compute hit points and normals where sphere_hit is True
        hit_points: Float[t.Tensor, "N 3"] = origin + pixel_directions * t_hit.unsqueeze(-1)
        normal_vectors: Float[t.Tensor, "N 3"] = F.normalize(hit_points - self.center, dim=1)

        # Update the record
        record.hit = sphere_hit
        record.t[sphere_hit] = t_hit[sphere_hit]
        record.point[sphere_hit] = hit_points[sphere_hit]
        record.normal[sphere_hit] = normal_vectors[sphere_hit]
        record.set_face_normal(pixel_directions, record.normal)

        return record
