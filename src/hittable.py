from abc import ABC, abstractmethod
from typing import List

import torch as t
from jaxtyping import Bool, Float, jaxtyped
from typeguard import typechecked as typechecker


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
        material_type = t.full(shape, -1, dtype=t.long, device=device)
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
        from config import device

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
