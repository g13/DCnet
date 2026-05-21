import json
from pathlib import Path
from unittest.mock import patch

from torchvision.transforms import transforms
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from data2_shared_cues import get_qclevr_dataloaders, qCLEVRDataset


class TinyQCLEVRDataset(Dataset):
    calls = []
    length = 8

    def __init__(self, *args, **kwargs):
        self.calls.append(kwargs)
        self.split = kwargs["split"]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return index


def make_loaders(dataset_length=8, **kwargs):
    TinyQCLEVRDataset.calls = []
    TinyQCLEVRDataset.length = dataset_length
    with patch("data2_shared_cues.qCLEVRDataset", TinyQCLEVRDataset):
        return get_qclevr_dataloaders(
            data_root="unused",
            assets_path="unused",
            train_batch_size=2,
            val_batch_size=2,
            resolution=(16, 16),
            **kwargs,
        )


def test_single_process_uses_random_train_sampler_and_sequential_val_sampler():
    train_loader, val_loader = make_loaders()

    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(val_loader.sampler, SequentialSampler)


def test_distributed_uses_train_distributed_sampler_with_expected_shuffle_flags():
    train_loader, val_loader = make_loaders(distributed=True, rank=1, world_size=2, seed=123)

    assert isinstance(train_loader.sampler, DistributedSampler)
    assert train_loader.sampler.rank == 1
    assert train_loader.sampler.num_replicas == 2
    assert train_loader.sampler.shuffle is True
    assert train_loader.sampler.seed == 123


def test_distributed_validation_sampler_covers_indices_once_without_padding():
    _, rank0_val_loader = make_loaders(dataset_length=5, distributed=True, rank=0, world_size=2)
    _, rank1_val_loader = make_loaders(dataset_length=5, distributed=True, rank=1, world_size=2)

    rank0_indices = list(rank0_val_loader.sampler)
    rank1_indices = list(rank1_val_loader.sampler)
    combined_indices = sorted(rank0_indices + rank1_indices)

    assert not isinstance(rank0_val_loader.sampler, DistributedSampler)
    assert not isinstance(rank1_val_loader.sampler, DistributedSampler)
    assert rank0_indices == [0, 2, 4]
    assert rank1_indices == [1, 3]
    assert combined_indices == [0, 1, 2, 3, 4]


def test_distributed_validation_sampler_allows_empty_ranks_without_duplicates():
    rank_indices = []
    for rank in range(5):
        _, val_loader = make_loaders(dataset_length=3, distributed=True, rank=rank, world_size=5)
        rank_indices.append(list(val_loader.sampler))

    combined_indices = sorted(index for indices in rank_indices for index in indices)

    assert rank_indices == [[0], [1], [2], [], []]
    assert combined_indices == [0, 1, 2]


def test_distributed_nonzero_rank_constructs_datasets_without_verbose_output():
    make_loaders(distributed=True, rank=1, world_size=2)

    assert [call["verbose"] for call in TinyQCLEVRDataset.calls] == [False, False]


def test_rank_zero_and_single_process_construct_datasets_with_verbose_output():
    make_loaders(distributed=True, rank=0, world_size=2)
    assert [call["verbose"] for call in TinyQCLEVRDataset.calls] == [True, True]

    make_loaders(distributed=False, rank=1, world_size=1)
    assert [call["verbose"] for call in TinyQCLEVRDataset.calls] == [True, True]


def test_old_layout_pool_fallback_does_not_print_when_verbose_false(tmp_path):
    data_root = tmp_path / "qclevr"
    scenes_dir = data_root / "train_color" / "scenes"
    images_dir = data_root / "train_color" / "images"
    scenes_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)
    image_path = images_dir / "scene.png"
    image_path.write_bytes(b"not loaded by construction")
    scene_path = scenes_dir / "scene.json"
    scene_path.write_text(
        json.dumps(
            {
                "image_filename": image_path.name,
                "cue": "red",
                "target_count": 1,
            }
        )
    )

    with patch("data2_shared_cues.multiprocessing.Pool", side_effect=OSError("no pool")), patch(
        "builtins.print"
    ) as print_:
        qCLEVRDataset(
            data_root=str(data_root),
            assets_path="unused",
            clevr_transforms=transforms.ToTensor(),
            split="train",
            mode="color",
            num_workers=2,
            verbose=False,
        )

    print_.assert_not_called()
