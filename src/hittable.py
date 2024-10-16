from abc import ABC, abstractmethod
from typing import List

import torch as t
from jaxtyping import Bool, Float, jaxtyped
from typeguard import typechecked as typechecker


@jaxtyped(typechecker=typechecker)
class HitRecord:
    """Class to register ray-object intersections."""

    def __init__(
        self,
        hit: Bool[t.Tensor, "..."],
        point: Float[t.Tensor, "... 3"],
        normal: Float[t.Tensor, "... 3"],
        t: Float[t.Tensor, "..."],
        front_face: Bool[t.Tensor, "..."] = None,
    ):
        self.hit: Bool[t.Tensor, "..."] = hit
        self.point: Float[t.Tensor, "... 3"] = point
        self.normal: Float[t.Tensor, "... 3"] = normal
        self.t: Float[t.Tensor, "..."] = t
        self.front_face: Bool[t.Tensor, "..."] = front_face

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
    def empty(shape: t.Size, device: t.device = "cpu") -> "HitRecord":
        """Creates an empty HitRecord with default values."""
        hit: Bool[t.Tensor, "..."] = t.full(shape, False, dtype=t.bool, device=device)
        point: Float[t.Tensor, "... 3"] = t.zeros((*shape, 3), dtype=t.float32, device=device)
        normal: Float[t.Tensor, "... 3"] = t.zeros((*shape, 3), dtype=t.float32, device=device)
        t_values: Float[t.Tensor, "..."] = t.full(shape, float("inf"), dtype=t.float32, device=device)
        front_face: Bool[t.Tensor, "..."] = t.full(shape, False, dtype=t.bool, device=device)
        return HitRecord(hit, point, normal, t_values, front_face)


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

    def __init__(self, objects: List[Hittable]):
        self.objects: List[Hittable] = objects

    def hit(
        self,
        pixel_rays: Float[t.Tensor, "N 3 2"],
        t_min: float,
        t_max: float,
    ) -> HitRecord:
        N: int = pixel_rays.shape[0]
        record: HitRecord = HitRecord.empty((N,), device=pixel_rays.device)
        closest_so_far: Float[t.Tensor, "N"] = t.full((N,), t_max, device=pixel_rays.device)

        for obj in self.objects:
            obj_record: HitRecord = obj.hit(pixel_rays, t_min, t_max)

            closer_mask: Bool[t.Tensor, "N"] = obj_record.hit & (obj_record.t < closest_so_far)
            closest_so_far = t.where(closer_mask, obj_record.t, closest_so_far)

            record.hit = record.hit | obj_record.hit
            record.point = t.where(closer_mask.unsqueeze(-1), obj_record.point, record.point)
            record.normal = t.where(closer_mask.unsqueeze(-1), obj_record.normal, record.normal)
            record.t = t.where(closer_mask, obj_record.t, record.t)

        record.set_face_normal(pixel_rays[:, :, 1], record.normal)
        return record
