import unittest
import warnings
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from subprocess import CalledProcessError
from unittest.mock import patch

import torch
import torch.nn as nn

from model import Conv2dEIRNN, Conv2dEIRNNCell
from model_fig4 import Conv2dEIRNNCell as Conv2dEIRNNCellFig4
from utils import (
    export_mermaid_diagram_assets,
    format_model_setup_report,
    get_git_commit_hash,
    make_model_diagram_filename,
    format_parameter_report,
    save_mermaid_diagram,
    save_mermaid_source,
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

    def build_feedback_modulated_dcnet(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", UserWarning)
            return Conv2dEIRNN(
                input_size=(8, 8),
                input_dim=3,
                h_pyr_dim=[2, 4],
                h_inter_dim=[1, 1],
                fb_dim=[3, 0],
                exc_kernel_size=[[3, 3], [3, 3]],
                inh_kernel_size=[[3, 3], [3, 3]],
                immediate_inhibition=True,
                num_layers=2,
                num_steps=1,
                num_classes=2,
                modulation=True,
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
                fb_adjacency=torch.tensor([[0, 0], [1, 0]]),
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
        self.assertIn("flowchart ", report)
        self.assertIn("Cue input", report)
        self.assertIn("Scene input", report)
        self.assertIn("Logits", report)

    def test_mermaid_area_params_report_cell_only_counts(self):
        model = self.build_hidden_modulated_dcnet()

        report = format_model_setup_report(model, include_mermaid=True)
        expected_area_params = sum(
            p.numel()
            for name, p in model.named_parameters()
            if name.startswith("layers.0.")
        )

        self.assertIn("cell params", report)
        self.assertIn(f">{expected_area_params:,}<", report)

    def test_mermaid_hidden_modulation_uses_state_resolution_for_cue_average(self):
        model = self.build_hidden_modulated_dcnet()

        report = format_model_setup_report(model, include_mermaid=True)

        self.assertIn("pool cue pyr", report)
        self.assertIn("[2, 8, 8] -&gt; [2]", report)
        self.assertIn("pool cue inter", report)
        self.assertIn("[1, 8, 8] -&gt; [1]", report)
        self.assertIn("rc target pyr", report)

    def test_mermaid_report_includes_modulation_and_feedback_detail_nodes(self):
        model = self.build_feedback_modulated_dcnet()

        report = format_model_setup_report(model, include_mermaid=True)
        modulation_params = sum(
            p.numel()
            for name, p in model.named_parameters()
            if name.startswith("modulations.0.")
        )
        feedback_params = sum(
            p.numel()
            for name, p in model.named_parameters()
            if name.startswith("fb_convs.fb_conv_1_0.")
        )

        self.assertIn("Mod A", report)
        self.assertIn("mode", report)
        self.assertIn("layer_output", report)
        self.assertIn("ff pool target pyr", report)
        self.assertIn("[2, 4, 4]", report)
        self.assertIn(f">{modulation_params:,}<", report)
        self.assertIn("fb B -&gt; A", report)
        self.assertIn("fb src pyr", report)
        self.assertIn("[4, 2, 2]", report)
        self.assertIn("fb out", report)
        self.assertIn("[3, 8, 8]", report)
        self.assertIn(f">{feedback_params:,}<", report)

    def test_mermaid_report_includes_box_highlighting_classes(self):
        model = self.build_feedback_modulated_dcnet()

        report = format_model_setup_report(model, include_mermaid=True)

        self.assertIn("classDef io ", report)
        self.assertIn("classDef area ", report)
        self.assertIn("classDef modulation ", report)
        self.assertIn("classDef feedback ", report)
        self.assertIn("classDef legend ", report)
        self.assertIn("class cue,scene io;", report)

    def test_mermaid_report_uses_table_labels_for_alignment(self):
        model = self.build_feedback_modulated_dcnet()

        report = format_model_setup_report(model, include_mermaid=True)

        self.assertIn("<div style='text-align:left'><table style='border-spacing:0'>", report)
        self.assertIn("white-space:nowrap", report)
        self.assertIn("rc state pyr/inter", report)
        self.assertIn("ff+rc+fb conv to pyr", report)

    def test_mermaid_report_uses_portrait_layout_for_wide_models(self):
        wide_model = self.build_feedback_modulated_dcnet()
        narrow_model = self.build_hidden_modulated_dcnet()

        wide_report = format_model_setup_report(wide_model, include_mermaid=True)
        narrow_report = format_model_setup_report(narrow_model, include_mermaid=True)

        self.assertIn("flowchart TB", wide_report)
        self.assertIn("flowchart LR", narrow_report)

    def test_mermaid_report_clarifies_trainable_dims_biases_and_taus(self):
        model = self.build_feedback_modulated_dcnet()

        report = format_model_setup_report(model, include_mermaid=True)

        self.assertIn("ff+rc+fb conv to pyr", report)
        self.assertIn("8 -&gt; 2 @ 3x3", report)
        self.assertIn("ff+rc+fb conv to inter", report)
        self.assertIn("rc conv inter-&gt;pyr", report)
        self.assertIn("tau pyr/inter", report)
        self.assertIn(">2 / 1<", report)
        self.assertIn("pool cue pyr", report)
        self.assertIn("fc h/w pyr", report)
        self.assertIn("fb resize", report)
        self.assertIn("1x1 conv 4 -&gt; 3", report)
        self.assertIn("fb out", report)
        self.assertIn("ff fc hidden", report)
        self.assertIn("bias", report)
        self.assertIn(">4<", report)
        self.assertIn("Legend", report)
        self.assertIn("feedforward / feedback / recurrent", report)
        self.assertIn("conv/fc/pool", report)
        self.assertIn("pyr/inter", report)

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

    def test_save_mermaid_source_writes_mmd_body(self):
        with TemporaryDirectory() as tmpdir:
            output_path = save_mermaid_source(
                "```mermaid\nflowchart LR\n  A --> B\n```",
                output_dir=tmpdir,
                when=date(2026, 3, 17),
                commit_hash="abc1234",
            )

            self.assertEqual(output_path.name, "flow_chart-2026-03-17-abc1234.mmd")
            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.read_text(encoding="utf-8"), "flowchart LR\n  A --> B")

    def test_export_mermaid_diagram_assets_writes_mmd_and_pdf(self):
        def fake_renderer(mermaid_text, output_path):
            self.assertTrue(mermaid_text.startswith("flowchart LR"))
            self.assertEqual(Path(output_path).suffix, ".pdf")
            Path(output_path).write_bytes(b"%PDF-unit-test")

        with TemporaryDirectory() as tmpdir:
            outputs = export_mermaid_diagram_assets(
                "```mermaid\nflowchart LR\n  A --> B\n```",
                output_dir=tmpdir,
                when=date(2026, 3, 17),
                commit_hash="abc1234",
                renderer=fake_renderer,
            )

            self.assertEqual(outputs["source_path"].name, "flow_chart-2026-03-17-abc1234.mmd")
            self.assertEqual(outputs["pdf_path"].name, "flow_chart-2026-03-17-abc1234.pdf")
            self.assertEqual(outputs["source_path"].read_text(encoding="utf-8"), "flowchart LR\n  A --> B")
            self.assertEqual(outputs["pdf_path"].read_bytes(), b"%PDF-unit-test")

    def test_get_git_commit_hash_returns_default_when_git_lookup_fails(self):
        with patch("utils.subprocess.check_output", side_effect=CalledProcessError(1, ["git"])):
            commit_hash = get_git_commit_hash(default="unknown")

        self.assertEqual(commit_hash, "unknown")

    def test_ei_cell_time_constants_are_per_channel(self):
        cell = Conv2dEIRNNCell(
            input_size=(8, 8),
            input_dim=3,
            h_pyr_dim=2,
            h_inter_dim=1,
            exc_kernel_size=(3, 3),
            inh_kernel_size=(3, 3),
            inh_rectify="pos",
            pool_kernel_size=(2, 2),
            pool_stride=(2, 2),
            pre_inh_activation="tanh",
            post_inh_activation=None,
        )

        self.assertEqual(tuple(cell.tau_pyr.shape), (1, 2, 1, 1))
        self.assertEqual(tuple(cell.tau_inter.shape), (1, 1, 1, 1))

    def test_fig4_ei_cell_time_constants_are_per_channel(self):
        cell = Conv2dEIRNNCellFig4(
            input_size=(8, 8),
            input_dim=3,
            h_pyr_dim=2,
            h_inter_dim=1,
            exc_kernel_size=(3, 3),
            inh_kernel_size=(3, 3),
            inh_rectify="pos",
            pool_kernel_size=(2, 2),
            pool_stride=(2, 2),
            pre_inh_activation="tanh",
            post_inh_activation=None,
        )

        self.assertEqual(tuple(cell.tau_pyr.shape), (1, 2, 1, 1))
        self.assertEqual(tuple(cell.tau_inter.shape), (1, 1, 1, 1))


if __name__ == "__main__":
    unittest.main()
