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
        hit: Bool[t.Tensor, "h w"],
        point: Float[t.Tensor, "h w 3"],
        normal: Float[t.Tensor, "h w 3"],
        t: Float[t.Tensor, "h w"],
        front_face: Bool[t.Tensor, "h w"] = None,
    ):
        self.hit = hit
        self.point = point
        self.normal = normal
        self.t = t
        self.front_face = front_face

    @jaxtyped(typechecker=typechecker)
    def set_face_normal(
        self, ray_direction: Float[t.Tensor, "h w 3"], outward_normal: Float[t.Tensor, "h w 3"]
    ) -> None:
        """Determines whether the hit is from the outside or inside."""
        self.front_face = (ray_direction * outward_normal).sum(dim=2) < 0
        self.normal = t.where(self.front_face.unsqueeze(-1), outward_normal, -outward_normal)

    @staticmethod
    @jaxtyped(typechecker=typechecker)
    def empty(h: int, w: int, device: t.device = "cpu") -> "HitRecord":
        """Creates an empty HitRecord with default values."""
        hit = t.full((h, w), False, dtype=t.bool, device=device)
        point = t.zeros((h, w, 3), dtype=t.float32, device=device)
        normal = t.zeros((h, w, 3), dtype=t.float32, device=device)
        t_values = t.full((h, w), float("inf"), dtype=t.float32, device=device)
        front_face = t.full((h, w), False, dtype=t.bool, device=device)
        return HitRecord(hit, point, normal, t_values, front_face)


@jaxtyped(typechecker=typechecker)
class Hittable(ABC):
    """Abstract class for hittable objects."""

    @abstractmethod
    @jaxtyped(typechecker=typechecker)
    def hit(self, pixel_rays: Float[t.Tensor, "h w 3 2"], t_min: float, t_max: float) -> HitRecord:
        pass


@jaxtyped(typechecker=typechecker)
class HittableList(Hittable):
    def __init__(self, objects: List[Hittable]):
        self.objects: List[Hittable] = objects
        self.hit_record: None | HitRecord = None

    @jaxtyped(typechecker=typechecker)
    def hit(self, pixel_rays: Float[t.Tensor, "h w 3 2"], t_min: float, t_max: float) -> HitRecord:
        h, w = pixel_rays.shape[:2]
        record = HitRecord.empty(h, w, device=pixel_rays.device)
        closest_so_far: Float[t.Tensor, "h w"] = t.full_like(record.t, t_max)

        for obj in self.objects:
            # Call the hit function of the object
            obj.hit(pixel_rays, t_min, t_max)

            # Create a mask where the current object is closer than any previous hit
            closer_mask: Bool[t.Tensor, "h w"] = obj.hit_record.hit & (obj.hit_record.t < closest_so_far)
            closest_so_far = t.where(closer_mask, obj.hit_record.t, closest_so_far)

            # Update the main hit record where the current object is closer
            record.hit = record.hit | obj.hit_record.hit
            record.point = t.where(closer_mask.unsqueeze(-1), obj.hit_record.point, record.point)
            record.normal = t.where(closer_mask.unsqueeze(-1), obj.hit_record.normal, record.normal)
            record.t = t.where(closer_mask, obj.hit_record.t, record.t)

        record.set_face_normal(pixel_rays[:, :, :, 1], record.normal)
        self.hit_record = record
        return record
