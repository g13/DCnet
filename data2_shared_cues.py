import glob
import json
import multiprocessing
import os
from typing import Callable, Optional

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import transforms
from utils import rescale, seed_worker


def draw_shape(image_size, shape_type, center, size=None, radius=None, color=(255, 255, 255)):
    image = Image.new("RGB", image_size, "gray")
    draw = ImageDraw.Draw(image)
    if shape_type == "cylinder":
        x1_top, y1_top = center[0] - size[0] // 2, center[1] + size[1] // 2 - 25
        x2_top, y2_top = center[0] + size[0] // 2, center[1] + size[1] // 2 + 25
        draw.ellipse([x1_top, y1_top, x2_top, y2_top], fill=color)
        x1_body, y1_body = center[0] - size[0] // 2, center[1] - size[1] // 2
        x2_body, y2_body = center[0] + size[0] // 2, center[1] + size[1] // 2
        draw.rectangle([x1_body, y1_body, x2_body, y2_body], fill=color)
        x1_bottom, y1_bottom = center[0] - size[0] // 2, center[1] - size[1] // 2 - 25
        x2_bottom, y2_bottom = center[0] + size[0] // 2, center[1] - size[1] // 2 + 25
        draw.ellipse([x1_bottom, y1_bottom, x2_bottom, y2_bottom], fill=color)
    elif shape_type == "cube":
        if size is None:
            raise ValueError("Size must be provided for cube.")
        x1, y1 = center[0] - size // 2, center[1] - size // 2
        x2, y2 = center[0] + size // 2, center[1] + size // 2
        draw.rectangle([x1, y1, x2, y2], fill=color)
    elif shape_type == "sphere":
        if radius is None:
            raise ValueError("Radius must be provided for sphere.")
        x1, y1 = center[0] - radius, center[1] - radius
        x2, y2 = center[0] + radius, center[1] + radius
        draw.ellipse([x1, y1, x2, y2], fill=color)
    else:
        raise ValueError("Invalid shape_type. Use cylinder, cube, or sphere.")
    return image


class qCLEVRDataset(Dataset):
    """
    Dataset for the shared-qCLEVR format.

    Preferred folder layout:
        data_root/
          train/images/*.png
          train/scenes/*.json
          valid/images/*.png
          valid/scenes/*.json
          cues/*.png                    # optional; can also pass assets_path

    Each scene JSON should contain:
        cue_type: color | shape | conjunction
        cue: red | cube | cube_red
        target_count: int

    Backward compatibility:
        If data_root/<split>/... does not exist, this loader falls back to the
        old layout data_root/<split>_<mode>/images and scenes.
    """

    def __init__(
        self,
        data_root: str,
        assets_path: str,
        clevr_transforms: Callable,
        return_images: bool = False,
        split: str = "train",
        holdout: list = [],
        mode: str = "color",
        primitive: bool = False,
        num_workers: int = 0,
        verbose: bool = True,
    ):
        super().__init__()
        self.data_root = data_root
        self.assets_path = assets_path
        self.clevr_transforms = clevr_transforms
        self.return_images = return_images
        self.mode = mode
        self.holdout = holdout
        self.split = split
        self.primitive = primitive
        self.num_workers = num_workers
        self.verbose = verbose

        assert os.path.exists(self.data_root), f"Path {self.data_root} does not exist"
        assert self.split in ("train", "valid", "test")

        self._modes = ["color", "shape", "conjunction"] if self.mode == "every" else [self.mode]

        self.color_dict = {
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "red": (173, 35, 35),
            "green": (29, 105, 20),
            "blue": (42, 75, 215),
            "yellow": (255, 238, 51),
            "purple": (129, 38, 192),
            "pink": (255, 192, 203),
            "orange": (255, 69, 0),
            "gray": (87, 87, 87),
            "brown": (129, 74, 25),
            "teal": (0, 128, 128),
            "navy": (0, 0, 128),
            "maroon": (128, 0, 0),
            "olive": (128, 128, 0),
            "cyan": (41, 208, 208),
        }
        self.shape_list = ["cylinder", "cube", "sphere"]

        # Use cue images from assets_path. If assets_path does not exist, try data_root/cues.
        if not assets_path or not os.path.isdir(assets_path):
            candidate = os.path.join(data_root, "cues")
            if os.path.isdir(candidate):
                self.assets_path = candidate
        self.cue_assets = self._load_cue_assets(self.assets_path)

        self.shared_layout = self._has_shared_layout()
        if self.verbose:
            print("*** Holding out: {}".format(self.holdout))
            print("*** Mode: {}".format(self.mode))
            print("*** Shared layout: {}".format(self.shared_layout))
            print("*** Cue assets path: {}".format(self.assets_path))

        self.files, self.cues, self.counts, self.modes = self.get_files()
        assert len(self.files) != 0, "Something about the config results in an empty dataset!"

    def _has_shared_layout(self):
        return (
            os.path.isdir(os.path.join(self.data_root, self.split, "images"))
            and os.path.isdir(os.path.join(self.data_root, self.split, "scenes"))
        )

    def _load_cue_assets(self, assets_path):
        assets = {}
        if assets_path and os.path.isdir(assets_path):
            for path in glob.glob(os.path.join(assets_path, "*.png")):
                key = os.path.splitext(os.path.basename(path))[0]
                assets[key] = path
        return assets

    def _open_asset(self, key):
        if key in self.cue_assets:
            return Image.open(self.cue_assets[key]).convert("RGB")
        raise KeyError(f"Cue asset {key}.png not found in {self.assets_path}")

    def _cue_key(self, mode, cue):
        if mode == "color":
            # Pure color patch generated as red.png, blue.png, ...
            return str(cue)
        if mode == "shape":
            # Shape cue rendered as fixed orange shape: cube_orange.png, etc.
            return f"{cue}_orange"
        if mode == "conjunction":
            if isinstance(cue, (list, tuple)):
                return f"{cue[0]}_{cue[1]}"
            return str(cue)
        raise NotImplementedError(mode)

    def _fallback_color_patch(self, image_size, color_name):
        if color_name not in self.color_dict:
            raise KeyError(f"Unknown color {color_name} and no cue asset found.")
        img = Image.new("RGB", image_size, self.color_dict[color_name])
        return img

    def _fallback_shape(self, image_size, shape, color=(255, 255, 255)):
        sz = 100
        if shape == "cylinder":
            sz = (50, 100)
        return draw_shape(
            image_size,
            shape,
            (image_size[0] / 2, image_size[1] / 2),
            size=sz,
            radius=100,
            color=color,
        )

    def _get_cue_image(self, img, mode, cue):
        key = self._cue_key(mode, cue)
        try:
            return self._open_asset(key)
        except KeyError:
            # Fallbacks keep old experiments runnable, but the new recommended
            # workflow is to generate cue PNGs with render_cues_full_patch.py.
            if mode == "color":
                return self._fallback_color_patch(img.size, cue)
            if mode == "shape":
                return self._fallback_shape(img.size, cue, color=(255, 255, 255))
            if mode == "conjunction":
                if isinstance(cue, str):
                    shape, color_name = cue.split("_", 1)
                else:
                    shape, color_name = cue[0], cue[1]
                return self._fallback_shape(img.size, shape, color=self.color_dict[color_name])
            raise

    def _passes_holdout(self, split, cue_key):
        if split == "train":
            return cue_key not in self.holdout
        return len(self.holdout) == 0 or cue_key in self.holdout

    def get_file_shared(self, scene_path):
        with open(scene_path, "r") as f:
            x = json.load(f)

        mode = x.get("cue_type", None)
        if mode is None:
            # Some older JSONs may use mode instead of cue_type.
            mode = x.get("mode", None)
        if mode not in self._modes:
            return None, None, None, None

        cue = x.get("cue")
        if mode == "conjunction":
            if "cue_shape" in x and "cue_color" in x:
                cue = f"{x['cue_shape']}_{x['cue_color']}"
            elif isinstance(cue, (list, tuple)):
                cue = f"{cue[0]}_{cue[1]}"
        cue_key = self._cue_key(mode, cue)
        if not self._passes_holdout(self.split, cue_key):
            return None, None, None, None

        image_path = os.path.join(self.data_root, self.split, "images", x["image_filename"])
        assert os.path.exists(image_path), f"{image_path} does not exist"
        return image_path, cue, int(x["target_count"]), mode

    def get_file_old_layout(self, _mode, scene_path):
        with open(scene_path, "r") as f:
            x = json.load(f)
            cue = x["cue"]
            cue_key = cue
            if _mode == "conjunction":
                if isinstance(cue, (list, tuple)):
                    cue_key = "{}_{}".format(cue[0], cue[1])
                else:
                    cue_key = str(cue)
            if not self._passes_holdout(self.split, cue_key):
                return None, None, None, None
            image_path = os.path.join(self.data_root, f"{self.split}_{_mode}", "images", x["image_filename"])
            assert os.path.exists(image_path), f"{image_path} does not exist"
            return image_path, cue, int(x["target_count"]), _mode

    def get_files(self):
        paths, cues, counts, modes = [], [], [], []

        if self.shared_layout:
            spath = os.path.join(self.data_root, self.split, "scenes")
            scene_paths = sorted(glob.glob(os.path.join(spath, "*.json")))
            for scene_path in scene_paths:
                path, cue, count, mode = self.get_file_shared(scene_path)
                if path is not None:
                    paths.append(path)
                    cues.append(cue)
                    counts.append(count)
                    modes.append(mode)
            return paths, cues, counts, modes

        # Backward-compatible old layout: train_color, train_shape, train_conjunction.
        pool = None
        use_multiprocessing = self.num_workers > 1
        if use_multiprocessing:
            try:
                pool = multiprocessing.Pool(self.num_workers)
            except (OSError, PermissionError) as exc:
                if self.verbose:
                    print(f"Falling back to single-process file scan because multiprocessing is unavailable: {exc}")
                use_multiprocessing = False

        for _mode in self._modes:
            spath = os.path.join(self.data_root, f"{self.split}_{_mode}", "scenes")
            scene_paths = sorted(glob.glob(os.path.join(spath, "*.json")))
            if use_multiprocessing:
                results = pool.starmap(self.get_file_old_layout, zip([_mode] * len(scene_paths), scene_paths))
            else:
                results = [self.get_file_old_layout(_mode, x) for x in scene_paths]
            for path, cue, count, mode in results:
                if path is not None:
                    paths.append(path)
                    cues.append(cue)
                    counts.append(count)
                    modes.append(mode)

        if pool is not None:
            pool.close()
            pool.join()
        return paths, cues, counts, modes

    def __getitem__(self, index: int):
        image_path = self.files[index]
        cue_str = self.cues[index]
        label = self.counts[index]
        mode = self.modes[index]

        img = Image.open(image_path).convert("RGB")
        cue = self._get_cue_image(img, mode, cue_str)

        if self.return_images:
            return (
                self.clevr_transforms(cue),
                self.clevr_transforms(img),
                label,
                image_path,
                mode,
            )
        return (
            self.clevr_transforms(cue),
            self.clevr_transforms(img),
            label,
            mode,
        )

    def __len__(self):
        return len(self.files)


class DistributedEvalSampler(Sampler[int]):
    def __init__(self, dataset: Dataset, rank: int, world_size: int):
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size")
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        return len(range(self.rank, len(self.dataset), self.world_size))


def get_qclevr_dataloaders(
    data_root: str,
    assets_path: str,
    train_batch_size: int,
    val_batch_size: int,
    resolution: tuple[int, int],
    holdout: list = [],
    mode: str = "color",
    primitive: bool = False,
    num_workers: int = 0,
    seed: Optional[int] = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    dataloader_kwargs = {}
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 4

    clevr_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(rescale),
            transforms.Resize(resolution),
        ]
    )
    verbose = not distributed or rank == 0

    train_dataset = qCLEVRDataset(
        data_root=data_root,
        assets_path=assets_path,
        clevr_transforms=clevr_transforms,
        split="train",
        holdout=holdout,
        mode=mode,
        primitive=primitive,
        num_workers=num_workers,
        verbose=verbose,
    )
    val_dataset = qCLEVRDataset(
        data_root=data_root,
        assets_path=assets_path,
        clevr_transforms=clevr_transforms,
        split="valid",
        holdout=holdout,
        mode=mode,
        primitive=primitive,
        num_workers=num_workers,
        verbose=verbose,
    )

    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed or 0,
        )
        val_sampler = DistributedEvalSampler(val_dataset, rank=rank, world_size=world_size)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if seed is not None else None,
        generator=torch.Generator().manual_seed(seed) if seed is not None else None,
        **dataloader_kwargs,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if seed is not None else None,
        generator=torch.Generator().manual_seed(seed) if seed is not None else None,
        **dataloader_kwargs,
    )
    return train_dataloader, val_dataloader
