import numpy as np
import torch as t
import torch.nn.functional as F
from jaxtyping import Float, jaxtyped
from PIL import Image
from typeguard import typechecked as typechecker

from hittable import HitRecord, Hittable


@jaxtyped(typechecker=typechecker)
class Camera:
    def __init__(
        self,
        origin: Float[t.Tensor, "3"] = t.Tensor([0.0, 0.0, 0.0]),
        focal_length: float = 1.0,
        aspect_ratio: float = 16.0 / 9.0,
        image_width: int = 400,
        viewport_height: float = 2.0,
        samples_per_pixel: int = 10,
    ):
        self.origin: Float[t.Tensor, "3"] = origin
        self.focal_length: float = focal_length
        self.samples_per_pixel: int = samples_per_pixel

        self.aspect_ratio: float = aspect_ratio
        self.image_width: int = image_width
        self.image_height: int = max(1, int(image_width / aspect_ratio))

        self.viewport_height: float = viewport_height
        self.viewport_width: float = viewport_height * (image_width / self.image_height)
        h, w = self.image_height, self.image_width,

        # x-axis
        left_border: float = -self.viewport_width / 2
        width_pixel_delta: float = self.viewport_width / image_width
        pixels_x: Float[t.Tensor, "w"] = t.linspace(
            left_border + width_pixel_delta / 2, -left_border - width_pixel_delta / 2, image_width
        )

        # y-axis
        top_border: float = viewport_height / 2
        height_pixel_delta: float = viewport_height / self.image_height
        pixels_y: Float[t.Tensor, "h"] = t.linspace(
            -top_border + height_pixel_delta / 2, top_border - height_pixel_delta / 2, self.image_height
        )

        # z-axis
        pixels_z: Float[t.Tensor, "1"] = t.Tensor([-focal_length])

        # Generate coordinate grids
        grid_x, grid_y, grid_z = t.meshgrid(pixels_x, pixels_y, pixels_z, indexing="ij")
        viewport_pixels = t.stack([grid_x, grid_y, grid_z], dim=-1)
        self.viewport_pixels = viewport_pixels.permute(1, 0, 2, 3).reshape(self.image_height, image_width, 3)
        self.pixel_rays: Float[t.Tensor, "h w 3 2"] = t.stack(
            [origin.expand(self.image_height, image_width, 3), self.viewport_pixels - origin], dim=-1
        )
        self.pixel_rays = F.normalize(self.pixel_rays, dim=2)

    @jaxtyped(typechecker=typechecker)
    def ray_color(
        self, background_colors: Float[t.Tensor, "h w 3"], hit_record: HitRecord
    ) -> Float[t.Tensor, "h w 3"]:

        background_colors[hit_record.hit] = ((hit_record.normal[hit_record.hit] + 1) * 0.5 * 255).to(
            dtype=background_colors.dtype
        )

        return background_colors

    @jaxtyped(typechecker=typechecker)
    def render(self, world: Hittable) -> Image:

        # Background color gradient
        white: t.Tensor = t.tensor([1.0, 1.0, 1.0])
        light_blue: t.Tensor = t.tensor([0.5, 0.7, 1.0])
        a: Float[t.Tensor, "h 1"] = t.linspace(0, 1, self.image_height).unsqueeze(1)
        background_colors: Float[t.Tensor, "h 3"] = a * light_blue + (1.0 - a) * white
        background_colors: Float[t.Tensor, "h w c"] = (
            background_colors.unsqueeze(1).expand(-1, self.image_width, -1) * 255
        )

        # Antialiasing
        for i in range(self.samples_per_pixel):
            hit_record: HitRecord = world.hit(self.pixel_rays, 0.0, float("inf"))

            # Ray color
            img = self.ray_color(background_colors, hit_record)
            array = img.cpu().numpy().astype(np.uint8)
            array = array[::-1, :, :]

        return Image.fromarray(array, mode="RGB")
