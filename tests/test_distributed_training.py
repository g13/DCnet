import os
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn as nn

import train
from utils import AttrDict


class ModuleWrapper:
    def __init__(self, module):
        self.module = module


class OrigModWrapper:
    def __init__(self, module):
        self._orig_mod = module


class FixedLogitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.0]))

    def forward(self, cue, mixture, all_timesteps=False):
        logits = torch.stack((mixture.reshape(-1), -mixture.reshape(-1)), dim=1)
        logits = logits + self.weight
        if all_timesteps:
            return [logits, logits]
        return logits


def make_config():
    return AttrDict(
        {
            "criterion": {"all_timesteps": False},
            "tqdm": False,
            "train": {
                "accumulation_steps": 1,
                "amp": False,
                "grad_clip": {"disable": True, "type": "norm", "value": 1.0},
                "log_freq": 100,
            },
        }
    )


def make_tqdm_config():
    config = make_config()
    config.tqdm = True
    return config


class DistributedTrainingTests(unittest.TestCase):
    def helper(self, name):
        helper = getattr(train, name, None)
        self.assertIsNotNone(helper)
        return helper

    def test_default_env_gives_disabled_single_process_context(self):
        distributed_context_from_env = self.helper("_distributed_context_from_env")
        DistributedContext = self.helper("DistributedContext")

        with patch.dict(os.environ, {}, clear=True):
            context = distributed_context_from_env()

        self.assertEqual(
            context,
            DistributedContext(enabled=False, local_rank=0, rank=0, world_size=1),
        )

    def test_torchrun_env_values_are_read(self):
        distributed_context_from_env = self.helper("_distributed_context_from_env")
        DistributedContext = self.helper("DistributedContext")

        with patch.dict(
            os.environ,
            {"LOCAL_RANK": "2", "RANK": "6", "WORLD_SIZE": "8"},
            clear=True,
        ):
            context = distributed_context_from_env()

        self.assertEqual(
            context,
            DistributedContext(enabled=True, local_rank=2, rank=6, world_size=8),
        )

    def test_is_main_process_only_for_rank_zero(self):
        is_main_process = self.helper("_is_main_process")
        DistributedContext = self.helper("DistributedContext")

        self.assertTrue(
            is_main_process(
                DistributedContext(enabled=True, local_rank=0, rank=0, world_size=2)
            )
        )
        self.assertFalse(
            is_main_process(
                DistributedContext(enabled=True, local_rank=1, rank=1, world_size=2)
            )
        )

    def test_unwrap_model_handles_nested_module_and_orig_mod_wrappers(self):
        unwrap_model = self.helper("_unwrap_model")
        model = nn.Linear(1, 1)
        wrapped = ModuleWrapper(OrigModWrapper(model))

        self.assertIs(unwrap_model(wrapped), model)

    def test_reduce_metrics_noops_when_context_disabled(self):
        reduce_metrics = self.helper("_reduce_metrics")
        DistributedContext = self.helper("DistributedContext")
        context = DistributedContext(enabled=False, local_rank=0, rank=0, world_size=1)

        reduced = reduce_metrics(3.5, 2, 4, torch.device("cpu"), context)

        self.assertEqual(reduced, (3.5, 2, 4))

    def test_reduce_metrics_all_reduces_sum_when_context_enabled(self):
        reduce_metrics = self.helper("_reduce_metrics")
        DistributedContext = self.helper("DistributedContext")
        context = DistributedContext(enabled=True, local_rank=0, rank=0, world_size=2)

        def fake_all_reduce(values, op):
            self.assertEqual(op, train.dist.ReduceOp.SUM)
            values += torch.tensor([4.0, 3.0, 6.0])

        with patch("train.dist.all_reduce", side_effect=fake_all_reduce) as all_reduce:
            reduced = reduce_metrics(3.5, 2, 4, torch.device("cpu"), context)

        all_reduce.assert_called_once()
        self.assertEqual(reduced, (7.5, 5, 10))

    def test_active_loader_import_uses_shared_cues_loader(self):
        self.assertEqual(train.get_qclevr_dataloaders.__module__, "data2_shared_cues")

    def test_setup_distributed_skips_process_group_when_disabled_and_uses_cpu(self):
        setup_distributed = self.helper("_setup_distributed")
        DistributedContext = self.helper("DistributedContext")
        context = DistributedContext(enabled=False, local_rank=0, rank=0, world_size=1)

        with patch("train.torch.cuda.is_available", return_value=False), patch(
            "train.dist.init_process_group"
        ) as init_process_group, patch("train.torch.cuda.set_device") as set_device:
            device = setup_distributed(context)

        init_process_group.assert_not_called()
        set_device.assert_not_called()
        self.assertEqual(device, torch.device("cpu"))

    def test_setup_distributed_initializes_gloo_on_cpu(self):
        setup_distributed = self.helper("_setup_distributed")
        DistributedContext = self.helper("DistributedContext")
        context = DistributedContext(enabled=True, local_rank=1, rank=1, world_size=2)

        with patch("train.torch.cuda.is_available", return_value=False), patch(
            "train.dist.init_process_group"
        ) as init_process_group, patch("train.torch.cuda.set_device") as set_device:
            device = setup_distributed(context)

        init_process_group.assert_called_once_with(backend="gloo")
        set_device.assert_not_called()
        self.assertEqual(device, torch.device("cpu"))

    def test_setup_distributed_initializes_nccl_and_selects_local_cuda_device(self):
        setup_distributed = self.helper("_setup_distributed")
        DistributedContext = self.helper("DistributedContext")
        context = DistributedContext(enabled=True, local_rank=2, rank=2, world_size=4)

        with patch("train.torch.cuda.is_available", return_value=True), patch(
            "train.dist.init_process_group"
        ) as init_process_group, patch("train.torch.cuda.set_device") as set_device:
            device = setup_distributed(context)

        init_process_group.assert_called_once_with(backend="nccl")
        set_device.assert_called_once_with(2)
        self.assertEqual(device, torch.device("cuda", 2))

    def test_device_for_context_uses_cuda_without_local_rank_when_not_distributed(self):
        device_for_context = self.helper("_device_for_context")
        DistributedContext = self.helper("DistributedContext")
        context = DistributedContext(enabled=False, local_rank=2, rank=0, world_size=1)

        with patch("train.torch.cuda.is_available", return_value=True):
            device = device_for_context(context)

        self.assertEqual(device, torch.device("cuda"))

    def test_wrap_distributed_model_skips_ddp_when_disabled(self):
        wrap_distributed_model = self.helper("_wrap_distributed_model")
        DistributedContext = self.helper("DistributedContext")
        context = DistributedContext(enabled=False, local_rank=0, rank=0, world_size=1)
        model = nn.Linear(1, 1)

        with patch("train.DistributedDataParallel") as ddp:
            wrapped = wrap_distributed_model(model, torch.device("cpu"), context)

        ddp.assert_not_called()
        self.assertIs(wrapped, model)

    def test_wrap_distributed_model_uses_cuda_device_ids(self):
        wrap_distributed_model = self.helper("_wrap_distributed_model")
        DistributedContext = self.helper("DistributedContext")
        context = DistributedContext(enabled=True, local_rank=3, rank=3, world_size=4)
        model = nn.Linear(1, 1)
        ddp_model = Mock()

        with patch("train.DistributedDataParallel", return_value=ddp_model) as ddp:
            wrapped = wrap_distributed_model(model, torch.device("cuda", 3), context)

        ddp.assert_called_once_with(
            model,
            device_ids=[3],
            output_device=3,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        self.assertIs(wrapped, ddp_model)

    def test_wrap_distributed_model_uses_cpu_ddp_without_device_ids(self):
        wrap_distributed_model = self.helper("_wrap_distributed_model")
        DistributedContext = self.helper("DistributedContext")
        context = DistributedContext(enabled=True, local_rank=0, rank=0, world_size=2)
        model = nn.Linear(1, 1)
        ddp_model = Mock()

        with patch("train.DistributedDataParallel", return_value=ddp_model) as ddp:
            wrapped = wrap_distributed_model(model, torch.device("cpu"), context)

        ddp.assert_called_once_with(
            model,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        self.assertIs(wrapped, ddp_model)

    def test_eval_iter_reports_sample_weighted_loss(self):
        model = FixedLogitModel()
        criterion = nn.CrossEntropyLoss()
        loader = [
            (torch.zeros(1, 1), torch.tensor([[2.0]]), torch.tensor([0])),
            (torch.zeros(3, 1), torch.full((3, 1), -2.0), torch.tensor([0, 0, 0])),
        ]

        loss, accuracy = train.eval_iter(
            make_config(),
            model,
            criterion,
            loader,
            lambda _: None,
            epoch=0,
            device=torch.device("cpu"),
        )

        expected_loss = (
            criterion(torch.tensor([[2.0, -2.0]]), torch.tensor([0])).item()
            + 3 * criterion(torch.tensor([[-2.0, 2.0]] * 3), torch.tensor([0, 0, 0])).item()
        ) / 4
        self.assertAlmostEqual(loss, expected_loss)
        self.assertEqual(accuracy, 0.25)

    def test_train_iter_reports_sample_weighted_loss(self):
        model = FixedLogitModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        criterion = nn.CrossEntropyLoss()
        loader = [
            (torch.zeros(1, 1), torch.tensor([[2.0]]), torch.tensor([0])),
            (torch.zeros(3, 1), torch.full((3, 1), -2.0), torch.tensor([0, 0, 0])),
        ]

        loss, accuracy = train.train_iter(
            make_config(),
            model,
            optimizer,
            None,
            criterion,
            loader,
            lambda _: None,
            epoch=0,
            device=torch.device("cpu"),
        )

        expected_loss = (
            criterion(torch.tensor([[2.0, -2.0]]), torch.tensor([0])).item()
            + 3 * criterion(torch.tensor([[-2.0, 2.0]] * 3), torch.tensor([0, 0, 0])).item()
        ) / 4
        self.assertAlmostEqual(loss, expected_loss)
        self.assertEqual(accuracy, 0.25)

    def test_train_iter_disables_tqdm_for_non_main_process(self):
        DistributedContext = self.helper("DistributedContext")
        model = FixedLogitModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        criterion = nn.CrossEntropyLoss()
        loader = [(torch.zeros(1, 1), torch.tensor([[2.0]]), torch.tensor([0]))]
        context = DistributedContext(enabled=True, local_rank=1, rank=1, world_size=2)

        with patch("train.tqdm", side_effect=lambda iterable, **kwargs: iterable) as tqdm, patch(
            "train._reduce_metrics", side_effect=lambda loss, correct, total, device, context: (loss, correct, total)
        ):
            train.train_iter(
                make_tqdm_config(),
                model,
                optimizer,
                None,
                criterion,
                loader,
                lambda _: None,
                epoch=0,
                device=torch.device("cpu"),
                context=context,
            )

        self.assertTrue(tqdm.call_args.kwargs["disable"])


if __name__ == "__main__":
    unittest.main()
