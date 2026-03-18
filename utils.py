import os
import random
import shutil
import subprocess
import tempfile
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


def format_mermaid_model_diagram(model):
    model = getattr(model, "_orig_mod", model)

    if not hasattr(model, "layers") or not hasattr(model, "out_layer"):
        raise TypeError("Mermaid model diagram is only supported for Conv2dEIRNN-style models.")

    counts = summarize_parameter_counts(model)
    input_channels = model.layers[0].input_dim
    input_h, input_w = model.input_sizes[0]

    lines = ["```mermaid", "flowchart LR"]
    lines.append(
        f'    cue["Cue input<br/>[{input_channels}, {input_h}, {input_w}]<br/>visual cue image"]'
    )
    lines.append(
        f'    scene["Scene input<br/>[{input_channels}, {input_h}, {input_w}]<br/>search image"]'
    )
    lines.append(f'    cuepass["Cue phase<br/>T = {model.num_steps} recurrent steps<br/>stores cue activations"]')
    lines.append(f'    scenepass["Scene phase<br/>T = {model.num_steps} recurrent steps"]')

    for i, layer in enumerate(model.layers):
        area = chr(ord("A") + i)
        state_h, state_w = model.input_sizes[i]
        out_h, out_w = model.output_sizes[i]
        layer_params = _group_parameter_counts(
            model,
            (
                f"layers.{i}.",
                f"modulations.{i}.",
                f"modulations_inter.{i}.",
                f"pertubations.{i}.",
                f"pertubations_inter.{i}.",
            ),
        )
        fb_in_params = 0
        for name, param in model.named_parameters():
            if name.startswith("fb_convs.") and name.split(".")[1].endswith(f"_{i}"):
                fb_in_params += param.numel()
        total_block_params = layer_params + fb_in_params
        kernel_h, kernel_w = model.exc_kernel_sizes[i]
        pad_h = kernel_h // 2
        pad_w = kernel_w // 2
        lines.append(
            f'    area{i}["Area {area}<br/>state E/I: {model.h_pyr_dims[i]}/{model.h_inter_dims[i]} @ {state_h}x{state_w}<br/>kernel: {kernel_h}x{kernel_w}, pad: {pad_h}/{pad_w}<br/>pooled out: [{model.h_pyr_dims[i]}, {out_h}, {out_w}]<br/>params: {total_block_params:,}"]'
        )

    flatten_dim = model.h_pyr_dims[-1] * prod(model.output_sizes[-1])
    hidden_linear = model.out_layer[1]
    final_linear = model.out_layer[-1]
    hidden_params = hidden_linear.weight.numel() + hidden_linear.bias.numel()
    final_params = final_linear.weight.numel() + final_linear.bias.numel()

    lines.append(f'    flatten["Flatten<br/>{flatten_dim:,}"]')
    lines.append(
        f'    fc1["FC hidden<br/>{flatten_dim:,} -> {hidden_linear.out_features}<br/>params: {hidden_params:,}"]'
    )
    lines.append(
        f'    logits["Logits<br/>{hidden_linear.out_features} -> {final_linear.out_features}<br/>params: {final_params:,}"]'
    )
    lines.append(
        f'    totals["Totals<br/>backbone: {counts["backbone"]:,}<br/>readout: {counts["readout"]:,}<br/>total: {counts["total"]:,}"]'
    )

    scene_chain = " --> ".join(["scenepass", *[f"area{i}" for i in range(len(model.layers))], "flatten", "fc1", "logits", "totals"])
    lines.append("    cue --> cuepass")
    lines.append("    scene --> scenepass")
    lines.append(f"    {scene_chain}")
    for i in range(len(model.layers)):
        lines.append(f"    cuepass -. cue-conditioned modulation .-> area{i}")
    lines.append("```")
    return "\n".join(lines)


def format_model_setup_report(model, include_mermaid=False, readout_prefix="out_layer"):
    sections = [format_parameter_report(model, readout_prefix=readout_prefix)]
    if include_mermaid:
        sections.append("Model shape + parameter flow:")
        sections.append(format_mermaid_model_diagram(model))
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


def resolve_model_output_mode(
    show_model_diagram=False,
    save_model_diagram_png=False,
    output_model_structure_only=False,
):
    return {
        "structure_only": output_model_structure_only,
        "show_model_diagram": show_model_diagram or output_model_structure_only,
        "save_model_diagram_png": save_model_diagram_png or output_model_structure_only,
    }


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


def save_mermaid_diagram_png(
    mermaid_report,
    output_dir=".",
    when=None,
    commit_hash="unknown",
    renderer=None,
):
    return save_mermaid_diagram(
        mermaid_report,
        output_dir=output_dir,
        when=when,
        commit_hash=commit_hash,
        output_format="png",
        renderer=renderer,
    )


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
