import os
import random
import shutil
import subprocess
import tempfile
import html
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from math import prod

import numpy as np
import scipy
import scipy.interpolate
import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = AttrDict(v)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_activation_class(activation):
    if activation is None or activation == "identity":
        return nn.Identity
    if activation == "relu":
        return nn.ReLU
    elif activation == "tanh":
        return nn.Tanh
    elif activation == "sigmoid":
        return nn.Sigmoid
    elif activation == "softplus":
        return nn.Softplus
    elif activation == "softsign":
        return nn.Softsign
    elif activation == "elu":
        return nn.ELU
    elif activation == "selu":
        return nn.SELU
    elif activation == "gelu":
        return nn.GELU
    elif activation == "leaky_relu":
        return nn.LeakyReLU
    else:
        raise ValueError(f"Activation function {activation} not supported.")


def idx_1D_to_2D(x, m, n):
    """
    Convert a 1D index to a 2D index.

    Args:
        x (torch.Tensor): 1D index.

    Returns:
        torch.Tensor: 2D index.
    """
    return torch.stack((x // m, x % n))


def idx_2D_to_1D(x, m, n):
    """
    Convert a 2D index to a 1D index.

    Args:
        x (torch.Tensor): 2D index.

    Returns:
        torch.Tensor: 1D index.
    """
    return x[0] * n + x[1]


def print_mem_stats():
    f, t = torch.cuda.mem_get_info()
    print(f"Free/Total: {f/(1024**3):.2f}GB/{t/(1024**3):.2f}GB")


def count_parameters(model):
    total_params = 0
    for param in model.parameters():
        num_params = (
            param._nnz()
            if param.layout in (torch.sparse_coo, torch.sparse_csr, torch.sparse_csc)
            else param.numel()
        )
        total_params += num_params
    return total_params


def summarize_parameter_counts(model, readout_prefix="out_layer"):
    model = getattr(model, "_orig_mod", model)

    def parameter_numel(param):
        return (
            param._nnz()
            if param.layout in (torch.sparse_coo, torch.sparse_csr, torch.sparse_csc)
            else param.numel()
        )

    total = 0
    trainable = 0
    readout = 0
    prefix = f"{readout_prefix}."

    for name, param in model.named_parameters():
        num_params = parameter_numel(param)
        total += num_params
        if param.requires_grad:
            trainable += num_params
        if name.startswith(prefix):
            readout += num_params

    return {
        "total": total,
        "trainable": trainable,
        "readout": readout,
        "backbone": total - readout,
    }


def format_parameter_report(model, readout_prefix="out_layer"):
    counts = summarize_parameter_counts(model, readout_prefix=readout_prefix)
    return "\n".join(
        [
            "Model parameters:",
            f"  total: {counts['total']:,}",
            f"  trainable: {counts['trainable']:,}",
            f"  backbone: {counts['backbone']:,}",
            f"  readout: {counts['readout']:,}",
        ]
    )


def _group_parameter_counts(model, prefixes):
    model = getattr(model, "_orig_mod", model)
    if isinstance(prefixes, str):
        prefixes = (prefixes,)
    total = 0
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            total += param.numel()
    return total


def _format_feature_shape(channels, height, width):
    return f"[{channels}, {height}, {width}]"


def _format_mermaid_node_label(title, *details):
    rows = ["<div style='text-align:left'><table style='border-spacing:0'>"]
    rows.append(
        f"<tr><td colspan='3' style='white-space:nowrap'><b>{html.escape(title)}</b></td></tr>"
    )
    for detail in details:
        if not detail:
            continue
        if ": " in detail:
            key, value = detail.split(": ", 1)
            rows.append(
                f"<tr><td style='white-space:nowrap;padding-right:6px'>{html.escape(key)}</td><td style='white-space:nowrap;padding:0 4px'>:</td><td style='white-space:nowrap'>{html.escape(value)}</td></tr>"
            )
        else:
            rows.append(
                f"<tr><td colspan='3' style='white-space:nowrap'>{html.escape(detail)}</td></tr>"
            )
    rows.append("</table></div>")
    return "".join(rows)


def _append_mermaid_class(lines, class_name, nodes):
    if nodes:
        lines.append(f"    class {','.join(nodes)} {class_name};")


def _count_bias_parameters(module):
    return sum(
        param.numel()
        for name, param in module.named_parameters()
        if name.split(".")[-1] == "bias"
    )


def _format_conv_train_dims(conv):
    kernel_h, kernel_w = conv.kernel_size
    return f"{conv.in_channels} -> {conv.out_channels} @ {kernel_h}x{kernel_w}"


def _format_ff_rc_fb_label(use_fb):
    return "ff+rc+fb" if use_fb else "ff+rc"


def _estimate_mermaid_text_width(text):
    if not text:
        return 0
    return len(text)


def _estimate_mermaid_chain_width(model):
    node_widths = [
        max(
            _estimate_mermaid_text_width("Scene phase"),
            _estimate_mermaid_text_width(
                f"steps: T = {model.num_steps} recurrent steps"
            ),
        )
    ]

    for i, layer in enumerate(model.layers):
        state_h, state_w = model.input_sizes[i]
        out_h, out_w = model.output_sizes[i]
        ff_rc_fb_label = _format_ff_rc_fb_label(model.use_fb[i])
        details = [
            f"rc state pyr/inter: {model.h_pyr_dims[i]}/{model.h_inter_dims[i]} @ {state_h}x{state_w}",
            f"ff in: {_format_feature_shape(layer.input_dim, state_h, state_w)}",
            (
                f"fb in: {_format_feature_shape(model.fb_dims[i], state_h, state_w)}"
                if model.use_fb[i]
                else None
            ),
            f"ff pool out pyr: {_format_feature_shape(model.h_pyr_dims[i], out_h, out_w)}",
            f"{ff_rc_fb_label} conv to pyr: {_format_conv_train_dims(layer.conv_exc_pyr)}",
            (
                f"{ff_rc_fb_label} conv to inter: {_format_conv_train_dims(layer.conv_exc_inter)}"
                if layer.h_inter_dim > 0
                else None
            ),
            (
                f"rc conv inter->pyr: {_format_conv_train_dims(layer.conv_inh)}"
                if layer.h_inter_dim > 0
                else None
            ),
            f"bias: {_count_bias_parameters(layer):,}",
            (
                f"tau pyr/inter: {layer.tau_pyr.numel()} / {layer.tau_inter.numel()}"
                if layer.h_inter_dim > 0
                else f"tau pyr: {layer.tau_pyr.numel()}"
            ),
            f"rc cell params: {_group_parameter_counts(model, f'layers.{i}.'):,}",
        ]
        node_widths.append(
            max(
                [_estimate_mermaid_text_width(f"Area {chr(ord('A') + i)}")]
                + [_estimate_mermaid_text_width(detail) for detail in details if detail]
            )
        )

    flatten_dim = model.h_pyr_dims[-1] * prod(model.output_sizes[-1])
    hidden_linear = model.out_layer[1]
    final_linear = model.out_layer[-1]
    hidden_params = hidden_linear.weight.numel() + hidden_linear.bias.numel()
    final_params = final_linear.weight.numel() + final_linear.bias.numel()
    counts = summarize_parameter_counts(model)

    node_widths.extend(
        [
            max(_estimate_mermaid_text_width("Flatten"), _estimate_mermaid_text_width(f"{flatten_dim:,}")),
            max(
                _estimate_mermaid_text_width("ff fc hidden"),
                _estimate_mermaid_text_width(
                    f"ff fc: {flatten_dim:,} -> {hidden_linear.out_features}"
                ),
                _estimate_mermaid_text_width(f"bias: {hidden_linear.bias.numel():,}"),
                _estimate_mermaid_text_width(f"params: {hidden_params:,}"),
            ),
            max(
                _estimate_mermaid_text_width("ff fc Logits"),
                _estimate_mermaid_text_width(
                    f"ff fc: {hidden_linear.out_features} -> {final_linear.out_features}"
                ),
                _estimate_mermaid_text_width(f"bias: {final_linear.bias.numel():,}"),
                _estimate_mermaid_text_width(f"params: {final_params:,}"),
            ),
            max(
                _estimate_mermaid_text_width("Totals"),
                _estimate_mermaid_text_width(f"backbone: {counts['backbone']:,}"),
                _estimate_mermaid_text_width(f"readout: {counts['readout']:,}"),
                _estimate_mermaid_text_width(f"total: {counts['total']:,}"),
            ),
        ]
    )

    return sum(node_widths) + 10 * max(len(node_widths) - 1, 0)


def _choose_mermaid_flow_direction(model, layout="auto"):
    if layout == "landscape":
        return "LR"
    if layout == "portrait":
        return "TB"
    if layout != "auto":
        raise ValueError("layout must be 'auto', 'landscape', or 'portrait'.")

    return "TB" if _estimate_mermaid_chain_width(model) > 180 else "LR"


def format_mermaid_model_diagram(model, layout="auto"):
    model = getattr(model, "_orig_mod", model)

    if not hasattr(model, "layers") or not hasattr(model, "out_layer"):
        raise TypeError("Mermaid model diagram is only supported for Conv2dEIRNN-style models.")

    counts = summarize_parameter_counts(model)
    input_channels = model.layers[0].input_dim
    input_h, input_w = model.input_sizes[0]

    flow_direction = _choose_mermaid_flow_direction(model, layout=layout)
    lines = ["```mermaid", f"flowchart {flow_direction}"]
    class_nodes = {
        "io": ["cue", "scene"],
        "phase": ["cuepass", "scenepass"],
        "area": [],
        "modulation": [],
        "feedback": [],
        "readout": ["flatten", "fc1", "logits"],
        "totals": ["totals"],
        "legend": ["legend"],
    }
    lines.append(
        f'    cue["{_format_mermaid_node_label("Cue input", f"shape: {_format_feature_shape(input_channels, input_h, input_w)}", "role: visual cue image")}"]'
    )
    lines.append(
        f'    scene["{_format_mermaid_node_label("Scene input", f"shape: {_format_feature_shape(input_channels, input_h, input_w)}", "role: search image")}"]'
    )
    lines.append(
        f'    cuepass["{_format_mermaid_node_label("Cue phase", f"steps: T = {model.num_steps} recurrent steps", "stores: cue activations")}"]'
    )
    lines.append(
        f'    scenepass["{_format_mermaid_node_label("Scene phase", f"steps: T = {model.num_steps} recurrent steps")}"]'
    )

    for i, layer in enumerate(model.layers):
        area = chr(ord("A") + i)
        state_h, state_w = model.input_sizes[i]
        out_h, out_w = model.output_sizes[i]
        layer_params = _group_parameter_counts(model, f"layers.{i}.")
        fb_input = (
            f"fb in: {_format_feature_shape(model.fb_dims[i], state_h, state_w)}"
            if model.use_fb[i]
            else None
        )
        tau_line = (
            f"tau pyr/inter: {layer.tau_pyr.numel()} / {layer.tau_inter.numel()}"
            if layer.h_inter_dim > 0
            else f"tau pyr: {layer.tau_pyr.numel()}"
        )
        ff_rc_fb_label = _format_ff_rc_fb_label(model.use_fb[i])
        lines.append(
            f'    area{i}["{_format_mermaid_node_label(f"Area {area}", f"rc state pyr/inter: {model.h_pyr_dims[i]}/{model.h_inter_dims[i]} @ {state_h}x{state_w}", f"ff in: {_format_feature_shape(layer.input_dim, state_h, state_w)}", fb_input, f"ff pool out pyr: {_format_feature_shape(model.h_pyr_dims[i], out_h, out_w)}", f"{ff_rc_fb_label} conv to pyr: {_format_conv_train_dims(layer.conv_exc_pyr)}", f"{ff_rc_fb_label} conv to inter: {_format_conv_train_dims(layer.conv_exc_inter)}" if layer.h_inter_dim > 0 else None, f"rc conv inter->pyr: {_format_conv_train_dims(layer.conv_inh)}" if layer.h_inter_dim > 0 else None, f"bias: {_count_bias_parameters(layer):,}", tau_line, f"rc cell params: {layer_params:,}")}"]'
        )
        class_nodes["area"].append(f"area{i}")

        if getattr(model, "modulation", False):
            mod_name = f"mod{i}"
            if model.modulation_on == "hidden":
                mod_module = model.modulations[i]
                mod_params = sum(param.numel() for param in mod_module.parameters())
                mod_inter_module = model.modulations_inter[i]
                mod_inter_params = sum(
                    param.numel() for param in mod_inter_module.parameters()
                )
                mod_inter_shape = (
                    f"rc target inter: {_format_feature_shape(model.h_inter_dims[i], state_h, state_w)}"
                    if model.h_inter_dims[i] > 0 and mod_inter_params > 0
                    else None
                )
                mod_train_line = (
                    f"fc h/w pyr: {mod_module.rank_one_vec_h.in_features} -> {mod_module.rank_one_vec_h.out_features}, {mod_module.rank_one_vec_w.in_features} -> {mod_module.rank_one_vec_w.out_features}"
                )
                mod_inter_train_line = (
                    f"fc h/w inter: {mod_inter_module.rank_one_vec_h.in_features} -> {mod_inter_module.rank_one_vec_h.out_features}, {mod_inter_module.rank_one_vec_w.in_features} -> {mod_inter_module.rank_one_vec_w.out_features}"
                    if mod_inter_shape is not None
                    else None
                )
                mod_param_line = (
                    f"params pyr/inter: {mod_params:,}/{mod_inter_params:,}"
                    if mod_inter_shape is not None
                    else f"params: {mod_params:,}"
                )
                mod_bias_line = (
                    f"bias pyr/inter: {_count_bias_parameters(mod_module):,}/{_count_bias_parameters(mod_inter_module):,}"
                    if mod_inter_shape is not None
                    else f"bias: {_count_bias_parameters(mod_module):,}"
                )
                lines.append(
                    f'    {mod_name}["{_format_mermaid_node_label(f"Mod {area}", "mode: hidden", f"pool cue pyr: {_format_feature_shape(model.h_pyr_dims[i], state_h, state_w)} -> [{model.h_pyr_dims[i]}]", f"pool cue inter: {_format_feature_shape(model.h_inter_dims[i], state_h, state_w)} -> [{model.h_inter_dims[i]}]" if mod_inter_shape is not None else None, f"rc target pyr: {_format_feature_shape(model.h_pyr_dims[i], state_h, state_w)}", mod_inter_shape, mod_train_line, mod_inter_train_line, mod_bias_line, mod_param_line)}"]'
                )
            else:
                mod_module = model.modulations[i]
                mod_params = sum(param.numel() for param in mod_module.parameters())
                lines.append(
                    f'    {mod_name}["{_format_mermaid_node_label(f"Mod {area}", "mode: layer_output", f"pool cue pyr: {_format_feature_shape(model.h_pyr_dims[i], out_h, out_w)} -> [{model.h_pyr_dims[i]}]", f"ff pool target pyr: {_format_feature_shape(model.h_pyr_dims[i], out_h, out_w)}", f"fc h/w pyr: {mod_module.rank_one_vec_h.in_features} -> {mod_module.rank_one_vec_h.out_features}, {mod_module.rank_one_vec_w.in_features} -> {mod_module.rank_one_vec_w.out_features}", f"bias: {_count_bias_parameters(mod_module):,}", f"params: {mod_params:,}")}"]'
                )
            class_nodes["modulation"].append(mod_name)

    flatten_dim = model.h_pyr_dims[-1] * prod(model.output_sizes[-1])
    hidden_linear = model.out_layer[1]
    final_linear = model.out_layer[-1]
    hidden_params = hidden_linear.weight.numel() + hidden_linear.bias.numel()
    final_params = final_linear.weight.numel() + final_linear.bias.numel()

    lines.append(f'    flatten["{_format_mermaid_node_label("Flatten", f"{flatten_dim:,}")}"]')
    lines.append(
        f'    fc1["{_format_mermaid_node_label("ff fc hidden", f"ff fc: {flatten_dim:,} -> {hidden_linear.out_features}", f"bias: {hidden_linear.bias.numel():,}", f"params: {hidden_params:,}")}"]'
    )
    lines.append(
        f'    logits["{_format_mermaid_node_label("ff fc Logits", f"ff fc: {hidden_linear.out_features} -> {final_linear.out_features}", f"bias: {final_linear.bias.numel():,}", f"params: {final_params:,}")}"]'
    )
    lines.append(
        f'    totals["{_format_mermaid_node_label("Totals", f"backbone: {counts["backbone"]:,}", f"readout: {counts["readout"]:,}", f"total: {counts["total"]:,}")}"]'
    )
    lines.append(
        f'    legend["{_format_mermaid_node_label("Legend", "ff/fb/rc: feedforward / feedback / recurrent", "conv/fc/pool: operator subtype", "pyr/inter: pyramidal / interneuron", "numeric arrows: learned input -> output dims for conv/fc", "bias: learned bias scalars", "tau: learned per-channel time constants", "params / rc cell params: include listed bias and tau counts")}"]'
    )

    scene_chain = " --> ".join(["scenepass", *[f"area{i}" for i in range(len(model.layers))], "flatten", "fc1", "logits", "totals"])
    lines.append("    cue --> cuepass")
    lines.append("    scene --> scenepass")
    lines.append(f"    {scene_chain}")

    if getattr(model, "modulation", False):
        for i in range(len(model.layers)):
            lines.append(f"    cuepass -. cue-conditioned .-> mod{i}")
            lines.append(f"    mod{i} -. applies .-> area{i}")

    if getattr(model, "fb_adjacency", None) is not None and hasattr(model, "fb_convs"):
        for source_i, targets in enumerate(model.fb_adjacency):
            for target_i in targets:
                fb_name = f"fb{source_i}_{target_i}"
                source_h, source_w = model.output_sizes[source_i]
                target_h, target_w = model.input_sizes[target_i]
                fb_params = _group_parameter_counts(
                    model, f"fb_convs.fb_conv_{source_i}_{target_i}."
                )
                source_area = chr(ord("A") + source_i)
                target_area = chr(ord("A") + target_i)
                lines.append(
                    f'    {fb_name}["{_format_mermaid_node_label(f"fb {source_area} -> {target_area}", f"fb src pyr: {_format_feature_shape(model.h_pyr_dims[source_i], source_h, source_w)}", f"fb resize: {_format_feature_shape(model.h_pyr_dims[source_i], target_h, target_w)}", f"fb conv: 1x1 conv {model.h_pyr_dims[source_i]} -> {model.fb_dims[target_i]}", f"fb out: {_format_feature_shape(model.fb_dims[target_i], target_h, target_w)}", f"bias: {model.fb_convs[f'fb_conv_{source_i}_{target_i}'][1].bias.numel():,}", f"params: {fb_params:,}")}"]'
                )
                lines.append(f"    area{source_i} -. feedback emit .-> {fb_name}")
                lines.append(f"    {fb_name} -. next step .-> area{target_i}")
                class_nodes["feedback"].append(fb_name)

    lines.append("    totals -. notation .-> legend")

    lines.append(
        "    classDef io fill:#e8f1ff,stroke:#2b6cb0,color:#0f172a,stroke-width:1.5px;"
    )
    lines.append(
        "    classDef phase fill:#fff4d6,stroke:#b7791f,color:#5c3b00,stroke-width:1.5px;"
    )
    lines.append(
        "    classDef area fill:#e8fbef,stroke:#2f855a,color:#123524,stroke-width:1.5px;"
    )
    lines.append(
        "    classDef modulation fill:#fff1f2,stroke:#c53030,color:#63171b,stroke-width:1.5px;"
    )
    lines.append(
        "    classDef feedback fill:#eef2ff,stroke:#4c51bf,color:#1a365d,stroke-width:1.5px,stroke-dasharray: 4 2;"
    )
    lines.append(
        "    classDef readout fill:#f5f3ff,stroke:#6b46c1,color:#2d1b69,stroke-width:1.5px;"
    )
    lines.append(
        "    classDef totals fill:#f3f4f6,stroke:#4b5563,color:#111827,stroke-width:1.5px;"
    )
    lines.append(
        "    classDef legend fill:#fffaf0,stroke:#b7791f,color:#5b3a00,stroke-width:1.5px;"
    )
    _append_mermaid_class(lines, "io", class_nodes["io"])
    _append_mermaid_class(lines, "phase", class_nodes["phase"])
    _append_mermaid_class(lines, "area", class_nodes["area"])
    _append_mermaid_class(lines, "modulation", class_nodes["modulation"])
    _append_mermaid_class(lines, "feedback", class_nodes["feedback"])
    _append_mermaid_class(lines, "readout", class_nodes["readout"])
    _append_mermaid_class(lines, "totals", class_nodes["totals"])
    _append_mermaid_class(lines, "legend", class_nodes["legend"])
    lines.append("```")
    return "\n".join(lines)


def format_model_setup_report(
    model,
    include_mermaid=False,
    readout_prefix="out_layer",
    mermaid_layout="auto",
):
    sections = [format_parameter_report(model, readout_prefix=readout_prefix)]
    if include_mermaid:
        sections.append("Model shape + parameter flow:")
        sections.append(
            format_mermaid_model_diagram(model, layout=mermaid_layout)
        )
    return "\n\n".join(sections)


def _extract_mermaid_body(mermaid_report):
    lines = mermaid_report.strip().splitlines()
    if lines and lines[0].strip() == "```mermaid":
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def make_model_diagram_filename(when=None, commit_hash="unknown", output_format="pdf"):
    when = when or date.today()
    return f"flow_chart-{when.isoformat()}-{commit_hash}.{output_format}"


def get_git_commit_hash(default="unknown"):
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return default


def _render_mermaid_with_mmdc(mermaid_report, output_path):
    mmdc_path = shutil.which("mmdc")
    if mmdc_path is None:
        raise RuntimeError("Could not find 'mmdc' in PATH; cannot render Mermaid diagram.")

    mermaid_body = _extract_mermaid_body(mermaid_report)
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "diagram.mmd"
        input_path.write_text(mermaid_body, encoding="utf-8")
        subprocess.run(
            [mmdc_path, "-i", str(input_path), "-o", str(output_path), "-b", "white"],
            check=True,
            capture_output=True,
            text=True,
        )


def save_mermaid_diagram(
    mermaid_report,
    output_dir=".",
    when=None,
    commit_hash="unknown",
    output_format="pdf",
    renderer=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / make_model_diagram_filename(
        when=when,
        commit_hash=commit_hash,
        output_format=output_format,
    )
    renderer = renderer or _render_mermaid_with_mmdc
    renderer(_extract_mermaid_body(mermaid_report), output_path)
    return output_path


def save_mermaid_source(
    mermaid_report,
    output_dir=".",
    when=None,
    commit_hash="unknown",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / make_model_diagram_filename(
        when=when,
        commit_hash=commit_hash,
        output_format="mmd",
    )
    output_path.write_text(_extract_mermaid_body(mermaid_report), encoding="utf-8")
    return output_path


def export_mermaid_diagram_assets(
    mermaid_report,
    output_dir=".",
    when=None,
    commit_hash="unknown",
    renderer=None,
):
    source_path = save_mermaid_source(
        mermaid_report,
        output_dir=output_dir,
        when=when,
        commit_hash=commit_hash,
    )
    pdf_path = save_mermaid_diagram(
        mermaid_report,
        output_dir=output_dir,
        when=when,
        commit_hash=commit_hash,
        output_format="pdf",
        renderer=renderer,
    )
    return {"source_path": source_path, "pdf_path": pdf_path}


def profile_fn(fn, kwargs, sort_by="cuda_time_total", row_limit=50):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        fn(kwargs)
    return prof.key_averages.table(sort_by=sort_by, row_limit=row_limit)


def r_theta_mp(data):
    tmp = np.exp(data[0] + 1j * data[1]) - 0.5
    return np.abs(tmp), np.angle(tmp)


def normalize_for_mp(indices, N_x=150, N_y=300, retina_radius=80):
    x, y = indices
    normalized_x = (1 - (x) / retina_radius) * 2.4 - 0.6
    normalized_y = ((y - N_y // 2) / np.sqrt(retina_radius**2.0)) * 3.5
    return normalized_x, normalized_y


def flatten_indices(indices, N_y=300):
    return indices[0] * N_y + indices[1]


def image2v1(
    image,
    retina_indices,
    image_top_corner=(4, 4),
    N_x=150,
    N_y=300,
    retina_radius=80,
):
    image_x, image_y = image.shape[1:]  # (C, H, W)
    img_ind = np.zeros((2, image_x, image_y))
    img_ind[0, :, :] = (
        np.tile(0 + np.arange(image_x), (image_y, 1)).T / image_x * image_top_corner[0]
    )
    img_ind[1, :, :] = (
        np.tile(np.arange(image_y) - image_y // 2, (image_x, 1))
        / image_y
        * image_top_corner[1]
        * 2
    )

    flat_img_ind = img_ind.reshape((2, image_x * image_y))

    normed_indices_retina = normalize_for_mp(retina_indices, N_x, N_y, retina_radius)
    r_indices, theta_indices = r_theta_mp(normed_indices_retina)

    v_field_x = r_indices * np.cos(theta_indices)
    v_field_y = r_indices * np.sin(theta_indices)

    device = image.device
    image = image.cpu().numpy()

    if len(image.shape) == 3:
        img_on_vfield = [
            scipy.interpolate.griddata(
                flat_img_ind.T,
                im.flatten(),
                np.array((v_field_x, v_field_y)).T,
            )
            for im in image
        ]
        img_on_vfield = np.stack(img_on_vfield)
    else:
        img_on_vfield = scipy.interpolate.griddata(
            flat_img_ind.T,
            image[0].flatten(),
            np.array((v_field_x, v_field_y)).T,
        )

    img_on_vfield = torch.from_numpy(img_on_vfield).to(device).float()
    img_on_vfield = torch.nan_to_num(img_on_vfield)
    return img_on_vfield


def compact(l):
    return list(filter(None, l))


def rescale(x):
    # qCLEVR images arrive from `ToTensor()` in [0, 1]. The training code uses [-1, 1]
    # inputs for both cues and scenes before they are passed into the recurrent model.
    return x * 2 - 1
