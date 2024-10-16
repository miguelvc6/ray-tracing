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
        self.hit_record: None | HitRecord = None

    @jaxtyped(typechecker=typechecker)
    def hit(self, pixel_rays: Float[t.Tensor, "h w 3 2"], t_min: float, t_max: float) -> HitRecord:
        h, w = pixel_rays.shape[:2]
        record = HitRecord.empty(h, w, device=pixel_rays.device)
        # Vector from ray origin to sphere center
        origin: Float[t.Tensor, "3"] = pixel_rays[:, :, :, 0][0][0]
        pixel_directions: Float[t.Tensor, "h w 3"] = pixel_rays[:, :, :, 1]
        oc: Float[t.Tensor, "3"] = origin - self.center

        # Quadratic coefficients
        a: float = 1.0  # Since pixel_rays is normalized
        b: Float[t.Tensor, "h w"] = 2.0 * t.einsum("h w c, c -> h w", pixel_directions, oc)
        c_scalar: float = oc.dot(oc) - self.radius**2
        discriminant: Float[t.Tensor, "h w"] = b.pow(2) - 4 * c_scalar

        # Initialize masks and t_hit
        sphere_hit: Bool[t.Tensor, "h w"] = discriminant >= 0
        t_hit: Float[t.Tensor, "h w"] = t.full_like(discriminant, float("inf"))

        # Compute square roots where discriminant is non-negative
        sqrt_discriminant: Float[t.Tensor, "h w"] = t.zeros_like(discriminant)
        sqrt_discriminant[sphere_hit] = t.sqrt(discriminant[sphere_hit])

        # Compute both roots
        t0: Float[t.Tensor, "h w"] = t.zeros_like(discriminant)
        t1: Float[t.Tensor, "h w"] = t.zeros_like(discriminant)
        t0[sphere_hit] = (-b[sphere_hit] - sqrt_discriminant[sphere_hit]) / (2.0 * a)
        t1[sphere_hit] = (-b[sphere_hit] + sqrt_discriminant[sphere_hit]) / (2.0 * a)

        # Masks for valid t0 and t1
        t0_valid: Bool[t.Tensor, "h w"] = (t0 > t_min) & (t0 < t_max)
        t1_valid: Bool[t.Tensor, "h w"] = (t1 > t_min) & (t1 < t_max)

        t_hit: Float[t.Tensor, "h w"] = t.where((t0_valid) & (t0 < t_hit), t0, t_hit)
        t_hit: Float[t.Tensor, "h w"] = t.where((t1_valid) & (t1 < t_hit), t1, t_hit)

        sphere_hit = sphere_hit & (t_hit < float("inf"))

        # Compute hit points and normals where sphere_hit is True
        hit_points: Float[t.Tensor, "h w 3"] = origin + pixel_directions * t_hit.unsqueeze(-1)
        normal_vectors: Float[t.Tensor, "h w 3"] = F.normalize(hit_points - self.center, dim=2)

        # Update the hit record
        record.hit = sphere_hit
        record.point = t.where(sphere_hit.unsqueeze(-1), hit_points, record.point)
        record.t = t.where(sphere_hit, t_hit, record.t)
        record.set_face_normal(pixel_directions, normal_vectors)
        self.hit_record = record

        return record