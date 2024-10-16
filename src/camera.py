import numpy as np
import torch as t
import torch.nn.functional as F
from jaxtyping import Bool, Float, jaxtyped
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
        h, w = self.image_height, self.image_width

        # x-axis
        left_border: float = -self.viewport_width / 2
        self.width_pixel_delta: float = self.viewport_width / image_width
        pixels_x: Float[t.Tensor, "w"] = t.linspace(
            left_border + self.width_pixel_delta / 2, -left_border - self.width_pixel_delta / 2, image_width
        )

        # y-axis
        top_border: float = viewport_height / 2
        self.height_pixel_delta: float = viewport_height / self.image_height
        pixels_y: Float[t.Tensor, "h"] = t.linspace(
            -top_border + self.height_pixel_delta / 2, top_border - self.height_pixel_delta / 2, self.image_height
        )

        # z-axis
        pixels_z: Float[t.Tensor, "1"] = t.Tensor([-focal_length])

        # Generate coordinate grids
        grid_x, grid_y, grid_z = t.meshgrid(pixels_x, pixels_y, pixels_z, indexing="ij")
        viewport_pixels: Float[t.Tensor, "h w 3"] = t.stack([grid_x, grid_y, grid_z], dim=-1)
        self.viewport_pixels = viewport_pixels.permute(1, 0, 2, 3).reshape(self.image_height, image_width, 3)

    @jaxtyped(typechecker=typechecker)
    def ray_color(
        self,
        background_colors: Float[t.Tensor, "sample h w 3"],
        hit_record: HitRecord,
    ) -> Float[t.Tensor, "sample h w 3"]:

        sample, h, w = background_colors.shape[:3]
        hit_mask: Bool[t.Tensor, "sample h w 1"] = hit_record.hit.view(sample, h, w).unsqueeze(-1)
        hit_normals: Float[t.Tensor, "sample h w 3"] = ((hit_record.normal.view(sample, h, w, 3) + 1) * 0.5 * 255).to(
            dtype=background_colors.dtype
        )

        colors: Float[t.Tensor, "sample h w 3"] = t.where(hit_mask, hit_normals, background_colors)
        return colors

    @jaxtyped(typechecker=typechecker)
    def render(self, world: Hittable) -> Image.Image:
        sample: int = self.samples_per_pixel
        h: int = self.image_height
        w: int = self.image_width

        # Background color gradient
        white: Float[t.Tensor, "3"] = t.tensor([1.0, 1.0, 1.0])
        light_blue: Float[t.Tensor, "3"] = t.tensor([0.5, 0.7, 1.0])
        a: Float[t.Tensor, "h 1"] = t.linspace(0, 1, h).unsqueeze(1)
        background_colors_single: Float[t.Tensor, "h 3"] = a * light_blue + (1.0 - a) * white
        background_colors: Float[t.Tensor, "sample h w 3"] = (
            background_colors_single.unsqueeze(0).unsqueeze(2).expand(sample, h, w, 3) * 255
        )

        # Antialiasing: Generate multiple samples per pixel
        noise: Float[t.Tensor, "sample h w 2"] = t.rand((sample, h, w, 2)) - 0.5
        scale: Float[t.Tensor, "2"] = t.tensor([self.width_pixel_delta, self.height_pixel_delta])
        noise = noise * scale.view(1, 1, 1, 2)

        # Adjust pixel positions
        sampled_pixels_xy: Float[t.Tensor, "sample h w 2"] = self.viewport_pixels[:, :, :2].unsqueeze(0) + noise
        sampled_pixels_z: Float[t.Tensor, "sample h w 1"] = (
            self.viewport_pixels[:, :, 2].unsqueeze(0).unsqueeze(-1).expand(sample, h, w, 1)
        )
        sampled_pixels: Float[t.Tensor, "sample h w 3"] = t.cat([sampled_pixels_xy, sampled_pixels_z], dim=-1)

        # Compute rays
        origin: Float[t.Tensor, "1 1 1 3"] = self.origin.view(1, 1, 1, 3)
        pixel_rays: Float[t.Tensor, "sample h w 3 2"] = t.stack(
            [origin.expand(sample, h, w, 3), sampled_pixels - self.origin], dim=-1
        )
        pixel_rays = F.normalize(pixel_rays, dim=3)

        # Reshape rays for processing
        pixel_rays_flat: Float[t.Tensor, "S 3 2"] = pixel_rays.view(-1, 3, 2)

        # Call world.hit with the reshaped rays
        hit_record: HitRecord = world.hit(pixel_rays_flat, 0.0, float("inf"))

        # Compute colors per sample
        colors: Float[t.Tensor, "sample h w 3"] = self.ray_color(background_colors, hit_record)

        # Average over samples
        img: Float[t.Tensor, "h w 3"] = colors.mean(dim=0)

        # Convert to uint8
        array: np.ndarray = img.cpu().numpy().astype(np.uint8)
        array = array[::-1, :, :]  # Flip vertically

        return Image.fromarray(array, mode="RGB")
