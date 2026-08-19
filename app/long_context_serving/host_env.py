# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Best-effort capture of GPU / CPU / OS info for campaign reports.

Each subsystem is captured in isolation so a partial failure (missing
`rocm-smi`, missing `lscpu`, etc.) still records what is available.
"""

from __future__ import annotations

import datetime as _dt
import functools
import hashlib
import inspect
import json
import os
import platform
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _run(cmd: List[str], timeout: float = 10.0) -> Optional[str]:
    if not shutil.which(cmd[0]):
        return None
    try:
        out = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.decode("utf-8", errors="replace")


# gfx target → marketing family, used when rocm-smi does not populate a
# marketing name (it reports "N/A" on the measured nodes). This is a *display*
# fallback only; the raw PCI ID is always kept in `card_model` so the report
# does not lie about what was probed.
_GFX_TARGET_FAMILY: Dict[str, str] = {
    "gfx950": "Instinct MI350-class (gfx950)",
    "gfx942": "Instinct MI300-class (gfx942)",
    "gfx90a": "Instinct MI200-class (gfx90a)",
    "gfx908": "Instinct MI100 (gfx908)",
}


def _normalize_pci_id(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip().lower()
    if not s:
        return ""
    if s.startswith("0x"):
        s = s[2:]
    s = s.lstrip("0") or "0"
    return f"0x{s}"


def _resolve_marketing_name(card: Dict[str, Any]) -> Optional[str]:
    """Return a human-readable name, or None if no resolution found.

    Order: rocm-smi `marketing_name` → gfx target family map.
    """
    raw = card.get("marketing_name")
    if raw is not None:
        s = str(raw).strip()
        if s and s.upper() not in {"N/A", "NONE"}:
            return s
    gfx = card.get("gfx_target")
    if gfx:
        gfx_s = str(gfx).strip().lower()
        if gfx_s in _GFX_TARGET_FAMILY:
            return _GFX_TARGET_FAMILY[gfx_s]
    return None


def _collect_gfx_targets_rocminfo() -> List[str]:
    """Return per-agent gfx target strings (e.g. `gfx950`) in rocminfo order.

    Only GPU agents are returned — CPU agents are filtered out by skipping
    any agent whose name is not `gfx*`.
    """
    raw = _run(["rocminfo"], timeout=15.0)
    if raw is None:
        return []
    targets: List[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if s.startswith("Name:") and "gfx" in s:
            tok = s.split(":", 1)[1].strip()
            # rocminfo prints both `gfx950` and amdhsa triples — only keep
            # the bare gfxNNN form.
            if tok.startswith("gfx") and "-" not in tok and ":" not in tok:
                targets.append(tok)
    return targets


def _collect_vbios_per_card() -> Dict[str, str]:
    raw = _run(["rocm-smi", "--showvbios", "--json"])
    if raw is None:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    out: Dict[str, str] = {}
    for key, fields in data.items():
        if not key.startswith("card") or not isinstance(fields, dict):
            continue
        v = fields.get("VBIOS version") or fields.get("VBIOS Version")
        if v:
            out[key] = str(v).strip()
    return out


def _collect_gpus_rocm() -> Dict[str, Any]:
    raw = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    if raw is None:
        return {"available": False, "reason": "rocm-smi not available"}
    try:
        data = json.loads(raw)
    except ValueError:
        return {"available": False, "reason": "rocm-smi json parse error"}
    vbios_map = _collect_vbios_per_card()
    gfx_targets = _collect_gfx_targets_rocminfo()
    cards = []
    for idx, (key, fields) in enumerate(sorted(data.items())):
        if not key.startswith("card"):
            continue
        gfx = gfx_targets[idx] if idx < len(gfx_targets) else None
        card: Dict[str, Any] = {
            "id": key,
            "marketing_name": fields.get("Card Series")
            or fields.get("Card Model")
            or fields.get("Card SKU"),
            "card_model": fields.get("Card Model"),
            "card_sku": fields.get("Card SKU"),
            "card_vendor": fields.get("Card Vendor"),
            "vram_total_b": fields.get("VRAM Total Memory (B)"),
            "vram_used_b": fields.get("VRAM Total Used Memory (B)"),
            "vbios_version": vbios_map.get(key),
            "gfx_target": gfx,
        }
        card["marketing_name_resolved"] = _resolve_marketing_name(card)
        cards.append(card)
    return {"available": True, "count": len(cards), "cards": cards}


def _collect_rocm_stack() -> Dict[str, Any]:
    """ROCm runtime version + amdgpu kernel driver version."""
    info: Dict[str, Any] = {}
    rocm_version_path = Path("/opt/rocm/.info/version")
    if rocm_version_path.is_file():
        try:
            info["rocm_version"] = rocm_version_path.read_text().strip()
        except OSError:
            pass
    driver_path = Path("/sys/module/amdgpu/version")
    if driver_path.is_file():
        try:
            info["amdgpu_driver_version"] = driver_path.read_text().strip()
        except OSError:
            pass
    info["available"] = bool(info)
    if not info["available"]:
        info["reason"] = "no /opt/rocm/.info/version or /sys/module/amdgpu/version"
    return info


def _collect_cpu_lscpu() -> Dict[str, Any]:
    raw = _run(["lscpu", "-J"])
    if raw is None:
        return {"available": False, "reason": "lscpu not available"}
    try:
        data = json.loads(raw)
    except ValueError:
        return {"available": False, "reason": "lscpu json parse error"}
    flat: Dict[str, str] = {}

    def _walk(items: Any) -> None:
        if isinstance(items, list):
            for item in items:
                _walk(item)
        elif isinstance(items, dict):
            field = items.get("field")
            data_val = items.get("data")
            if field is not None:
                key = field.rstrip(":").strip()
                flat[key] = data_val
            children = items.get("children")
            if children:
                _walk(children)

    _walk(data.get("lscpu", []))
    return {
        "available": True,
        "model_name": flat.get("Model name"),
        "vendor_id": flat.get("Vendor ID"),
        "architecture": flat.get("Architecture"),
        "cpus_logical": flat.get("CPU(s)"),
        "sockets": flat.get("Socket(s)"),
        "cores_per_socket": flat.get("Core(s) per socket"),
        "threads_per_core": flat.get("Thread(s) per core"),
        "numa_nodes": flat.get("NUMA node(s)"),
        "max_mhz": flat.get("CPU max MHz"),
    }


def _collect_os() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }
    os_release = Path("/etc/os-release")
    if os_release.is_file():
        try:
            for line in os_release.read_text().splitlines():
                if "=" not in line:
                    continue
                k, _, v = line.partition("=")
                v = v.strip().strip('"')
                if k == "PRETTY_NAME":
                    info["pretty_name"] = v
                elif k == "ID":
                    info["distro_id"] = v
                elif k == "VERSION_ID":
                    info["distro_version_id"] = v
        except OSError:
            pass
    return info


@functools.lru_cache(maxsize=1)
def collect_host_env() -> Dict[str, Any]:
    """Gather GPU / CPU / OS / ROCm info. Cached per process."""
    return {
        "collected_at_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "os": _collect_os(),
        "cpu": _collect_cpu_lscpu(),
        "gpus": _collect_gpus_rocm(),
        "rocm": _collect_rocm_stack(),
    }


def _fmt_bytes_gib(value: Any) -> Optional[str]:
    try:
        gib = int(value) / (1024 ** 3)
    except (TypeError, ValueError):
        return None
    return f"{gib:.1f} GiB"


def render_environment_section(host_env: Optional[Dict[str, Any]]) -> List[str]:
    """Render an `## Environment` markdown section for a host_env block.

    Shared between the campaign-level and per-run report builders so both
    surface the same fields. Old run_meta.json blobs that predate the new
    `rocm` / `vbios_version` / `gfx_target` fields degrade gracefully —
    missing fields are simply omitted rather than rendered as `unknown`.
    """
    out: List[str] = ["## Environment", ""]
    if not host_env:
        out.append(
            "- _No `host_env` recorded in this run's `run_meta.json` "
            "(predates host-env capture)._"
        )
        out.append("")
        return out

    out.append(f"- Host: `{host_env.get('hostname', 'unknown')}`")

    os_info = host_env.get("os") or {}
    pretty = os_info.get("pretty_name") or os_info.get("system") or "unknown"
    kernel = os_info.get("release") or "unknown"
    machine = os_info.get("machine") or ""
    out.append(
        f"- OS: {pretty} (kernel `{kernel}`{', ' + machine if machine else ''})"
    )

    rocm = host_env.get("rocm") or {}
    if rocm.get("available"):
        rocm_v = rocm.get("rocm_version") or "?"
        drv = rocm.get("amdgpu_driver_version") or "?"
        out.append(f"- ROCm: `{rocm_v}` (amdgpu driver `{drv}`)")

    cpu = host_env.get("cpu") or {}
    if cpu.get("available"):
        model = cpu.get("model_name") or "unknown"
        sockets = cpu.get("sockets") or "?"
        cores = cpu.get("cores_per_socket") or "?"
        threads = cpu.get("threads_per_core") or "?"
        logical = cpu.get("cpus_logical") or "?"
        numa = cpu.get("numa_nodes") or "?"
        out.append(
            f"- CPU: {model}, {sockets} socket(s) × {cores} cores × "
            f"{threads} threads = {logical} logical, NUMA {numa}"
        )
    elif cpu:
        out.append(f"- CPU: _{cpu.get('reason', 'unavailable')}_")

    gpus = host_env.get("gpus") or {}
    if gpus.get("available"):
        cards = gpus.get("cards") or []
        count = gpus.get("count") or len(cards)
        first = cards[0] if cards else {}
        # Prefer the resolved marketing name written at capture time; fall
        # back to in-place resolution for older run_meta.json blobs.
        marketing = first.get("marketing_name_resolved") or _resolve_marketing_name(first)
        pci = _normalize_pci_id(first.get("card_model")) or "unknown"
        gfx = first.get("gfx_target") or ""
        if marketing:
            label = f"{marketing} (`{pci}`{', ' + gfx if gfx else ''})"
        else:
            label = f"`{pci}`{' (' + gfx + ')' if gfx else ''}"
        vram_total = _fmt_bytes_gib(first.get("vram_total_b")) or "unknown"
        vram_used = _fmt_bytes_gib(first.get("vram_used_b"))
        if vram_used:
            try:
                free_b = int(first.get("vram_total_b")) - int(first.get("vram_used_b"))
                vram_free = _fmt_bytes_gib(free_b) or "?"
            except (TypeError, ValueError):
                vram_free = "?"
            vram_part = (
                f"{vram_total}/GPU total, {vram_free} free at capture "
                f"(used {vram_used})"
            )
        else:
            vram_part = f"{vram_total}/GPU"
        out.append(f"- GPU: {count} × {label}, {vram_part}")
        # VBIOS — collapsed if all cards share one version (typical), else
        # listed per-card so a mismatched board is visible.
        vbios = [str(c.get("vbios_version")) for c in cards if c.get("vbios_version")]
        if vbios:
            uniq = sorted(set(vbios))
            if len(uniq) == 1:
                out.append(f"- VBIOS (all cards): `{uniq[0]}`")
            else:
                per = ", ".join(
                    f"{c.get('id', '?')}=`{c.get('vbios_version', '?')}`"
                    for c in cards
                )
                out.append(f"- VBIOS (per card): {per}")
    elif gpus:
        out.append(f"- GPU: _{gpus.get('reason', 'unavailable')}_")

    if host_env.get("collected_at_utc"):
        out.append(f"- Captured: {host_env['collected_at_utc']}")
    out.append("")
    return out


# Files we md5 to identify which compiled MLA decode kernel is installed.
# The blob names are the AITER convention; the md5 is recorded in the run
# fingerprint so the report can name the exact decode kernel that shipped.
_MLA_KERNEL_BLOB_NAMES = (
    "mla_a8w8_qh16_qseqlen1_gqaratio16.co",
    "mla_dec_stage1_bf16_a16w16_subQ16_mqa16.co",
)

# Files we md5 to identify which compiled FMHA prefill kernel is installed.
# Both causal and non-causal _group variants are tracked because vLLM's MLA
# chunked-prefill loop dispatches to non-causal (`_group`) for K chunks before
# the Q range and causal (`_causal_group`) for the chunk that straddles it.
_FMHA_KERNEL_BLOB_NAMES = (
    "fwd_hd192_hd128_bf16_causal_group.co",
    "fwd_hd192_hd128_bf16_group.co",
)

# Legacy markers grep'd in aiter/mla.py source, recorded in the fingerprint
# for provenance. On the current vLLM v0.25.1 + AITER v0.1.19.post2 stack
# head padding is done vLLM-side, so this marker is normally absent; it is
# kept only so older runs remain comparable.
_MLA_PY_MARKERS = ("_head_pad_factor",)

# Env vars for the host-side chunked-K FMHA prefill dispatcher. Nothing in this
# tree sets them; they are read so that runs recorded on a stack that did have
# the dispatcher keep reporting whether it was active and at what chunk size.
_CHUNKED_DISPATCHER_ENV_KEYS = (
    "HFLC_FMHA_PREFILL_CHUNKED",
    "HFLC_FMHA_CHUNK_SIZE_K",
    "HFLC_FMHA_CHUNKED_DEBUG",
    "HFLC_FMHA_CHUNKED_SNAPSHOTS",
)


def _md5_of_file(path: Path, chunk: int = 1 << 20) -> Optional[str]:
    try:
        h = hashlib.md5()
        with path.open("rb") as fh:
            while True:
                buf = fh.read(chunk)
                if not buf:
                    break
                h.update(buf)
        return h.hexdigest()
    except OSError:
        return None


def _walk_for_blobs(root: Path, names: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    wanted = set(names)
    found: Dict[str, Dict[str, Any]] = {}
    if not root.exists():
        return found
    for path in root.rglob("*.co"):
        if path.name in wanted and path.name not in found:
            found[path.name] = {
                "path": str(path),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "md5": _md5_of_file(path),
            }
    return found


def _probe_aiter() -> Dict[str, Any]:
    info: Dict[str, Any] = {"available": False}
    try:
        import aiter  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - probe only
        info["reason"] = f"import aiter failed: {type(exc).__name__}: {exc}"
        return info
    info["available"] = True
    info["version"] = getattr(aiter, "__version__", None)
    info["package_file"] = getattr(aiter, "__file__", None)

    aiter_root = Path(info["package_file"]).parent if info["package_file"] else None
    if aiter_root is not None:
        info["package_dir"] = str(aiter_root)

    mla_src: Optional[Path] = None
    try:
        import aiter.mla as _mla  # type: ignore[import-not-found]
        src = inspect.getsourcefile(_mla) or getattr(_mla, "__file__", None)
        if src:
            mla_src = Path(src)
    except Exception as exc:  # noqa: BLE001
        info["mla_import_error"] = f"{type(exc).__name__}: {exc}"

    if mla_src is not None and mla_src.is_file():
        info["mla_py_path"] = str(mla_src)
        info["mla_py_size_bytes"] = mla_src.stat().st_size
        try:
            text = mla_src.read_text(errors="replace")
            info["mla_py_markers"] = {
                marker: text.count(marker) for marker in _MLA_PY_MARKERS
            }
        except OSError as exc:
            info["mla_py_markers_error"] = f"{type(exc).__name__}: {exc}"

    # Compiled .co blobs live in the sibling `aiter_meta` package, NOT under
    # the `aiter` package itself. The previous version of this collector
    # walked aiter_root and found nothing — every prior run_meta.json has an
    # empty mla_kernel_blobs because of that bug. Fixed here.
    aiter_meta_root = (
        aiter_root.parent / "aiter_meta" if aiter_root is not None else None
    )
    info["aiter_meta_dir"] = (
        str(aiter_meta_root) if aiter_meta_root is not None else None
    )
    info["mla_kernel_blobs"] = (
        _walk_for_blobs(aiter_meta_root, _MLA_KERNEL_BLOB_NAMES)
        if aiter_meta_root is not None and aiter_meta_root.exists()
        else {}
    )
    info["fmha_kernel_blobs"] = (
        _walk_for_blobs(aiter_meta_root, _FMHA_KERNEL_BLOB_NAMES)
        if aiter_meta_root is not None and aiter_meta_root.exists()
        else {}
    )

    # Build-time hints set by the Dockerfile or CI; harmless if absent.
    build_hints = {
        key: os.environ[key]
        for key in (
            "AITER_GIT_COMMIT",
            "AITER_GIT_BRANCH",
            "AITER_BUILD_TAG",
            "AITER_SOURCE_URL",
        )
        if key in os.environ
    }
    if build_hints:
        info["build_env_hints"] = build_hints
    return info


def _probe_vllm() -> Dict[str, Any]:
    info: Dict[str, Any] = {"available": False}
    try:
        import vllm  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        info["reason"] = f"import vllm failed: {type(exc).__name__}: {exc}"
        return info
    info["available"] = True
    info["version"] = getattr(vllm, "__version__", None)
    info["package_file"] = getattr(vllm, "__file__", None)
    if info["package_file"]:
        info["package_dir"] = str(Path(info["package_file"]).parent)
    return info


def _probe_chunked_dispatcher_env() -> Dict[str, str]:
    """Snapshot env vars that drive the host-side FMHA chunked-K dispatcher.

    Captured at run start by collect_kernel_fingerprint(); the consolidated
    report uses these to say whether the dispatcher was active for this run
    and, if so, with what chunk size. Default values are NOT filled in so
    the report can distinguish "not set" from "explicitly set to 0".
    """
    return {
        key: os.environ[key]
        for key in _CHUNKED_DISPATCHER_ENV_KEYS
        if key in os.environ
    }


def _probe_decode_dispatch_env() -> Dict[str, str]:
    keys = (
        "VLLM_USE_HIP_MLA_DECODE",
        "VLLM_HIP_MLA_KERNEL_DIR",
    )
    return {key: os.environ[key] for key in keys if key in os.environ}


@functools.lru_cache(maxsize=1)
def collect_mla_kernel_fingerprint() -> Dict[str, Any]:
    """Identify which MLA kernel(s) the installed aiter/vllm will dispatch to.

    Capturing the aiter version, mla.py marker count, and `.co` md5s lets the
    report tell readers which kernel actually ran.
    """
    return {
        "collected_at_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "aiter": _probe_aiter(),
        "vllm": _probe_vllm(),
        "decode_dispatch_env": _probe_decode_dispatch_env(),
        "chunked_dispatcher_env": _probe_chunked_dispatcher_env(),
    }


if __name__ == "__main__":
    payload = {
        "host_env": collect_host_env(),
        "mla_kernel_fingerprint": collect_mla_kernel_fingerprint(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
