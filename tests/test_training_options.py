import unittest
from contextlib import contextmanager
import inspect
from unittest.mock import patch

import torch
import torch.nn as nn

import train
from train import eval_iter, train_iter
from utils import AttrDict


class TinyCueSceneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 2, bias=False)

    def forward(self, cue, mixture, all_timesteps=False):
        logits = self.linear(mixture.reshape(mixture.shape[0], 1))
        if all_timesteps:
            return [logits, logits]
        return logits


class SingleParameterLogitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.0]))

    def forward(self, cue, mixture, all_timesteps=False):
        positive_logit = mixture.reshape(-1) * self.weight
        logits = torch.stack((positive_logit, torch.zeros_like(positive_logit)), dim=1)
        if all_timesteps:
            return [logits, logits]
        return logits


class MeanPositiveLogitLoss(nn.Module):
    def forward(self, outputs, labels):
        return outputs[:, 0].mean()


class CountingSGD(torch.optim.SGD):
    def __init__(self, params):
        super().__init__(params, lr=0.1)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure=closure)


class CountingScheduler:
    def __init__(self):
        self.step_count = 0

    def step(self):
        self.step_count += 1


class CountsOnlyDataset:
    counts = [0, 1, 1, 3]

    def __getitem__(self, index):
        raise AssertionError("debug summaries should use counts metadata")


class RaisingDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        raise AssertionError("debug summaries should not iterate shuffled samplers")


def make_config(accumulation_steps=1, amp=False, print_batch_progress=False, log_freq=100):
    return AttrDict(
        {
            "criterion": {"all_timesteps": False},
            "tqdm": False,
            "train": {
                "accumulation_steps": accumulation_steps,
                "amp": amp,
                "grad_clip": {"disable": True, "type": "norm", "value": 1.0},
                "log_freq": log_freq,
                "print_batch_progress": print_batch_progress,
            },
        }
    )


def make_batches(num_batches):
    batches = []
    for i in range(num_batches):
        cue = torch.zeros(1, 1)
        mixture = torch.tensor([[float(i + 1)]])
        label = torch.tensor([i % 2], dtype=torch.long)
        batches.append((cue, mixture, label))
    return batches


def make_shared_batches(num_batches):
    batches = []
    for i in range(num_batches):
        cue = torch.zeros(1, 1)
        mixture = torch.tensor([[float(i + 1)]])
        label = torch.tensor([i % 2], dtype=torch.long)
        mode = "train" if i % 2 == 0 else "eval"
        batches.append((cue, mixture, label, mode))
    return batches


def make_unequal_batches():
    return [
        (
            torch.zeros(2, 1),
            torch.tensor([[1.0], [3.0]]),
            torch.zeros(2, dtype=torch.long),
        ),
        (
            torch.zeros(1, 1),
            torch.tensor([[9.0]]),
            torch.zeros(1, dtype=torch.long),
        ),
    ]


class TrainingOptionsTests(unittest.TestCase):
    def test_label_counts_from_dataset_uses_counts_without_getitem(self):
        label_counts_from_dataset = getattr(train, "_label_counts_from_dataset", None)

        self.assertIsNotNone(label_counts_from_dataset)
        counts = label_counts_from_dataset(CountsOnlyDataset())

        torch.testing.assert_close(counts, torch.tensor([1, 2, 0, 1]))

    def test_label_counts_from_dataset_skips_dataset_without_counts(self):
        label_counts_from_dataset = getattr(train, "_label_counts_from_dataset", None)

        self.assertIsNotNone(label_counts_from_dataset)
        self.assertIsNone(label_counts_from_dataset(make_batches(1)))

    def test_first_batch_label_counts_accepts_shared_loader_batch(self):
        first_batch_label_counts = getattr(train, "_first_batch_label_counts", None)

        self.assertIsNotNone(first_batch_label_counts)
        counts = first_batch_label_counts(make_shared_batches(1))

        torch.testing.assert_close(counts, torch.tensor([1]))

    def test_first_batch_label_counts_skips_random_sampler_without_advancing_generator(self):
        first_batch_label_counts = getattr(train, "_first_batch_label_counts", None)
        generator = torch.Generator().manual_seed(123)
        loader = torch.utils.data.DataLoader(
            make_shared_batches(4), batch_size=2, shuffle=True, generator=generator
        )
        state_before = generator.get_state()

        self.assertIsNotNone(first_batch_label_counts)
        counts = first_batch_label_counts(loader)

        self.assertIsNone(counts)
        torch.testing.assert_close(generator.get_state(), state_before)

    def test_first_batch_label_counts_skips_shuffled_distributed_sampler_without_iterating(self):
        first_batch_label_counts = getattr(train, "_first_batch_label_counts", None)
        dataset = RaisingDataset()
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=2, rank=0, shuffle=True
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, sampler=sampler)

        self.assertIsNotNone(first_batch_label_counts)
        counts = first_batch_label_counts(loader)

        self.assertIsNone(counts)

    def test_optimizer_steps_per_epoch_rounds_up_for_partial_accumulation(self):
        optimizer_steps_per_epoch = getattr(train, "_optimizer_steps_per_epoch", None)

        self.assertIsNotNone(optimizer_steps_per_epoch)
        self.assertEqual(optimizer_steps_per_epoch(num_batches=5, accumulation_steps=2), 3)
        self.assertEqual(optimizer_steps_per_epoch(num_batches=4, accumulation_steps=2), 2)
        self.assertEqual(optimizer_steps_per_epoch(num_batches=4, accumulation_steps=0), 4)

    def test_build_scheduler_returns_none_for_disabled_scheduler_values(self):
        build_scheduler = getattr(train, "_build_scheduler", None)
        model = TinyCueSceneModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        self.assertIsNotNone(build_scheduler)
        for scheduler_fn in (None, "None", "null"):
            with self.subTest(scheduler_fn=scheduler_fn):
                config = AttrDict(
                    {
                        "scheduler": {"fn": scheduler_fn, "pct_start": 0.3},
                        "optimizer": {"lr": 0.1},
                        "train": {"epochs": 2, "accumulation_steps": 1},
                    }
                )

                self.assertIsNone(build_scheduler(config, optimizer, num_batches=3))

    def test_build_scheduler_builds_one_cycle_for_optimizer_steps(self):
        build_scheduler = getattr(train, "_build_scheduler", None)
        model = TinyCueSceneModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        config = AttrDict(
            {
                "scheduler": {"fn": "one_cycle", "pct_start": 0.25},
                "optimizer": {"lr": 0.1},
                "train": {"epochs": 2, "accumulation_steps": 2},
            }
        )
        sentinel = object()

        self.assertIsNotNone(build_scheduler)
        with patch("train.OneCycleLR", return_value=sentinel) as one_cycle:
            scheduler = build_scheduler(config, optimizer, num_batches=5)

        self.assertIs(scheduler, sentinel)
        one_cycle.assert_called_once_with(
            optimizer,
            max_lr=0.1,
            total_steps=2 * 3,
            pct_start=0.25,
        )

    def test_checkpointing_enabled_respects_disable_flag(self):
        checkpointing_enabled = getattr(train, "_checkpointing_enabled", None)

        self.assertIsNotNone(checkpointing_enabled)
        self.assertFalse(checkpointing_enabled(AttrDict({"checkpoint": {"disable": True}})))
        self.assertTrue(checkpointing_enabled(AttrDict({"checkpoint": {"disable": False}})))

    def test_train_guards_checkpoint_side_effects_with_checkpoint_disable(self):
        checkpointing_enabled = getattr(train, "_checkpointing_enabled", None)

        self.assertIsNotNone(checkpointing_enabled)
        source = inspect.getsource(train.train)
        self.assertIn("_checkpointing_enabled(config)", source)
        self.assertGreaterEqual(source.count("checkpointing_enabled"), 3)

    def test_train_iter_accumulates_gradients_before_optimizer_step(self):
        model = TinyCueSceneModel()
        optimizer = CountingSGD(model.parameters())
        scheduler = CountingScheduler()

        train_iter(
            make_config(accumulation_steps=2),
            model,
            optimizer,
            scheduler,
            nn.CrossEntropyLoss(),
            make_batches(2),
            lambda _: None,
            epoch=0,
            device=torch.device("cpu"),
        )

        self.assertEqual(optimizer.step_count, 1)
        self.assertEqual(scheduler.step_count, 1)

    def test_train_iter_prints_batch_progress_when_enabled(self):
        model = TinyCueSceneModel()
        optimizer = CountingSGD(model.parameters())

        with patch("builtins.print") as print_:
            train_iter(
                make_config(print_batch_progress=True, log_freq=1),
                model,
                optimizer,
                None,
                nn.CrossEntropyLoss(),
                make_batches(1),
                lambda _: None,
                epoch=3,
                device=torch.device("cpu"),
            )

        progress_lines = [
            call.args[0]
            for call in print_.call_args_list
            if call.args and "Batch: 1/1" in call.args[0]
        ]
        self.assertEqual(len(progress_lines), 1)
        self.assertIn("Epoch: 3", progress_lines[0])

    def test_train_iter_weights_short_final_microbatch_by_sample_count(self):
        model = SingleParameterLogitModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        train_iter(
            make_config(accumulation_steps=2),
            model,
            optimizer,
            None,
            MeanPositiveLogitLoss(),
            make_unequal_batches(),
            lambda _: None,
            epoch=0,
            device=torch.device("cpu"),
        )

        expected_weight = torch.tensor([-0.1 * ((1.0 + 3.0 + 9.0) / 3.0)])
        torch.testing.assert_close(model.weight.detach(), expected_weight)

    def test_train_iter_does_not_store_loss_tensors_across_accumulation_group(self):
        source = inspect.getsource(train.train_iter)

        self.assertNotIn("accumulation_group.append((loss", source)
        self.assertNotIn("group_loss = sum(", source)

    def test_train_iter_accepts_shared_loader_batch_and_steps_optimizer(self):
        model = TinyCueSceneModel()
        optimizer = CountingSGD(model.parameters())

        train_iter(
            make_config(),
            model,
            optimizer,
            None,
            nn.CrossEntropyLoss(),
            make_shared_batches(1),
            lambda _: None,
            epoch=0,
            device=torch.device("cpu"),
        )

        self.assertEqual(optimizer.step_count, 1)

    def test_eval_iter_accepts_shared_loader_batch(self):
        model = TinyCueSceneModel()

        loss, accuracy = eval_iter(
            make_config(),
            model,
            nn.CrossEntropyLoss(),
            make_shared_batches(1),
            lambda _: None,
            epoch=0,
            device=torch.device("cpu"),
        )

        self.assertIsInstance(loss, float)
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_train_iter_uses_autocast_context_from_amp_config(self):
        autocast_calls = []

        @contextmanager
        def fake_autocast(*, device_type, enabled):
            autocast_calls.append((device_type, enabled))
            yield

        model = TinyCueSceneModel()
        optimizer = CountingSGD(model.parameters())

        with patch("train.torch.amp.autocast", side_effect=fake_autocast):
            train_iter(
                make_config(amp=True),
                model,
                optimizer,
                None,
                nn.CrossEntropyLoss(),
                make_batches(1),
                lambda _: None,
                epoch=0,
                device=torch.device("cpu"),
            )

        self.assertEqual(autocast_calls, [("cpu", False)])

    def test_eval_iter_uses_autocast_context_from_amp_config(self):
        autocast_calls = []

        @contextmanager
        def fake_autocast(*, device_type, enabled):
            autocast_calls.append((device_type, enabled))
            yield

        model = TinyCueSceneModel()

        with patch("train.torch.amp.autocast", side_effect=fake_autocast):
            eval_iter(
                make_config(amp=True),
                model,
                nn.CrossEntropyLoss(),
                make_batches(1),
                lambda _: None,
                epoch=0,
                device=torch.device("cpu"),
            )

        self.assertEqual(autocast_calls, [("cpu", False)])


if __name__ == "__main__":
    unittest.main()
