import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from subprocess import CalledProcessError
from unittest.mock import patch

import torch.nn as nn

from model import Conv2dEIRNN
from utils import (
    format_model_setup_report,
    get_git_commit_hash,
    make_model_diagram_filename,
    format_parameter_report,
    resolve_model_output_mode,
    save_mermaid_diagram,
    save_mermaid_diagram_png,
    summarize_parameter_counts,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 3)
        self.out_layer = nn.Linear(3, 2)


class ParameterReportingTests(unittest.TestCase):
    def build_tiny_dcnet(self):
        return Conv2dEIRNN(
            input_size=(8, 8),
            input_dim=3,
            h_pyr_dim=[2, 4],
            h_inter_dim=[1, 1],
            fb_dim=[0, 0],
            exc_kernel_size=[[3, 3], [3, 3]],
            inh_kernel_size=[[3, 3], [3, 3]],
            immediate_inhibition=True,
            num_layers=2,
            num_steps=1,
            num_classes=2,
            modulation=False,
            modulation_type="lr",
            modulation_on="layer_output",
            modulation_timestep="all",
            pertubation=False,
            pertubation_type="lr",
            pertubation_on="hidden",
            pertubation_timestep=0,
            layer_time_delay=False,
            exc_rectify=None,
            inh_rectify="pos",
            flush_hidden=True,
            hidden_init_mode="zeros",
            fb_init_mode="zeros",
            out_init_mode="zeros",
            fb_adjacency=None,
            pool_kernel_size=[2, 2],
            pool_stride=[2, 2],
            bias=True,
            dropout=0.0,
            pre_inh_activation="tanh",
            post_inh_activation=None,
            fc_dim=4,
        )

    def build_hidden_modulated_dcnet(self):
        return Conv2dEIRNN(
            input_size=(8, 8),
            input_dim=3,
            h_pyr_dim=[2],
            h_inter_dim=[1],
            fb_dim=[0],
            exc_kernel_size=[[3, 3]],
            inh_kernel_size=[[3, 3]],
            immediate_inhibition=True,
            num_layers=1,
            num_steps=1,
            num_classes=2,
            modulation=True,
            modulation_type="lr",
            modulation_on="hidden",
            modulation_timestep="all",
            pertubation=False,
            pertubation_type="lr",
            pertubation_on="hidden",
            pertubation_timestep=0,
            layer_time_delay=False,
            exc_rectify=None,
            inh_rectify="pos",
            flush_hidden=True,
            hidden_init_mode="zeros",
            fb_init_mode="zeros",
            out_init_mode="zeros",
            fb_adjacency=None,
            pool_kernel_size=[2, 2],
            pool_stride=[2, 2],
            bias=True,
            dropout=0.0,
            pre_inh_activation="tanh",
            post_inh_activation=None,
            fc_dim=4,
        )

    def test_summarize_parameter_counts_separates_readout(self):
        model = TinyModel()

        counts = summarize_parameter_counts(model)

        self.assertEqual(counts["total"], 23)
        self.assertEqual(counts["backbone"], 15)
        self.assertEqual(counts["readout"], 8)
        self.assertEqual(counts["trainable"], 23)

    def test_format_parameter_report_includes_named_sections(self):
        model = TinyModel()

        report = format_parameter_report(model)

        self.assertIn("Model parameters:", report)
        self.assertIn("total: 23", report)
        self.assertIn("backbone: 15", report)
        self.assertIn("readout: 8", report)

    def test_format_model_setup_report_omits_mermaid_when_disabled(self):
        model = TinyModel()

        report = format_model_setup_report(model, include_mermaid=False)

        self.assertIn("Model parameters:", report)
        self.assertNotIn("```mermaid", report)

    def test_format_model_setup_report_includes_mermaid_when_enabled(self):
        model = self.build_tiny_dcnet()

        report = format_model_setup_report(model, include_mermaid=True)

        self.assertIn("```mermaid", report)
        self.assertIn("flowchart LR", report)
        self.assertIn("Cue input", report)
        self.assertIn("Scene input", report)
        self.assertIn("Logits", report)

    def test_mermaid_area_params_include_hidden_modulation_modules(self):
        model = self.build_hidden_modulated_dcnet()

        report = format_model_setup_report(model, include_mermaid=True)
        expected_area_params = sum(
            p.numel()
            for name, p in model.named_parameters()
            if name.startswith("layers.0.")
            or name.startswith("modulations.0.")
            or name.startswith("modulations_inter.0.")
        )

        self.assertIn(f"params: {expected_area_params:,}", report)

    def test_make_model_diagram_filename_uses_date_and_commit(self):
        filename = make_model_diagram_filename(date(2026, 3, 17), "abc1234")

        self.assertEqual(filename, "flow_chart-2026-03-17-abc1234.pdf")

    def test_save_mermaid_diagram_writes_pdf_by_default(self):
        def fake_renderer(mermaid_text, output_path):
            self.assertTrue(mermaid_text.startswith("flowchart LR"))
            self.assertEqual(Path(output_path).suffix, ".pdf")
            Path(output_path).write_bytes(b"%PDF-unit-test")

        with TemporaryDirectory() as tmpdir:
            output_path = save_mermaid_diagram(
                "```mermaid\nflowchart LR\n  A --> B\n```",
                output_dir=tmpdir,
                when=date(2026, 3, 17),
                commit_hash="abc1234",
                renderer=fake_renderer,
            )

            self.assertEqual(output_path.name, "flow_chart-2026-03-17-abc1234.pdf")
            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.read_bytes(), b"%PDF-unit-test")

    def test_save_mermaid_diagram_png_keeps_png_output(self):
        def fake_renderer(mermaid_text, output_path):
            self.assertTrue(mermaid_text.startswith("flowchart LR"))
            self.assertEqual(Path(output_path).suffix, ".png")
            Path(output_path).write_bytes(b"\x89PNG\r\n\x1a\nunit-test")

        with TemporaryDirectory() as tmpdir:
            output_path = save_mermaid_diagram_png(
                "```mermaid\nflowchart LR\n  A --> B\n```",
                output_dir=tmpdir,
                when=date(2026, 3, 17),
                commit_hash="abc1234",
                renderer=fake_renderer,
            )

            self.assertEqual(output_path.name, "flow_chart-2026-03-17-abc1234.png")
            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.read_bytes(), b"\x89PNG\r\n\x1a\nunit-test")

    def test_get_git_commit_hash_returns_default_when_git_lookup_fails(self):
        with patch("utils.subprocess.check_output", side_effect=CalledProcessError(1, ["git"])):
            commit_hash = get_git_commit_hash(default="unknown")

        self.assertEqual(commit_hash, "unknown")

    def test_resolve_model_output_mode_defaults_to_existing_switches(self):
        mode = resolve_model_output_mode(
            show_model_diagram=False,
            save_model_diagram_png=True,
            output_model_structure_only=False,
        )

        self.assertFalse(mode["structure_only"])
        self.assertFalse(mode["show_model_diagram"])
        self.assertTrue(mode["save_model_diagram_png"])

    def test_resolve_model_output_mode_forces_diagram_outputs_in_structure_only_mode(self):
        mode = resolve_model_output_mode(
            show_model_diagram=False,
            save_model_diagram_png=False,
            output_model_structure_only=True,
        )

        self.assertTrue(mode["structure_only"])
        self.assertTrue(mode["show_model_diagram"])
        self.assertTrue(mode["save_model_diagram_png"])


if __name__ == "__main__":
    unittest.main()
