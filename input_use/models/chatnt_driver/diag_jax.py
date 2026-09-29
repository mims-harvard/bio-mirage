#!/usr/bin/env python
"""Loads the released ChatNT jax parameters (params.joblib) and writes a per-leaf .npy copy with a
sha256 manifest, which smoke_jax.py reads.

Part of the checks of the ChatNT model setup. It also records the dtype of every parameter leaf, and
whether loading changes jax's bfloat16 type or type promotion, in jax_params_diag.json.

    python input_use/models/chatnt_driver/diag_jax.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chatnt_common import ROOT, RESULTS  # noqa: E402  (sets HF/XDG env)

out = {"steps": []}


def step(name, fn):
    t = time.time()
    try:
        r = fn()
        out["steps"].append({"name": name, "ok": True, "seconds": time.time() - t, "result": r})
        print(f"[ok] {name}: {r}", flush=True)
        return r
    except Exception as e:  # noqa: BLE001
        out["steps"].append({"name": name, "ok": False, "seconds": time.time() - t,
                             "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]})
        print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)
        return None


import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import ml_dtypes  # noqa: E402
import numpy  # noqa: E402

out["versions"] = {"jax": jax.__version__, "ml_dtypes": ml_dtypes.__version__, "numpy": numpy.__version__}
step("PRNGKey_before_load", lambda: str(jax.random.PRNGKey(0)))
step("bf16_identity_before", lambda: {"ml_dtypes_bf16": repr(ml_dtypes.bfloat16), "jnp_bf16": repr(jnp.bfloat16),
                                      "same": jnp.bfloat16 is ml_dtypes.bfloat16,
                                      "np_dtype_hash": hash(np.dtype(ml_dtypes.bfloat16))})

from nucleotide_transformer.chatNT.params import download_ckpt  # noqa: E402

params = step("download_ckpt", lambda: (globals().__setitem__("P", download_ckpt()), "loaded")[1])
if params is None:
    json.dump(out, open(os.path.join(RESULTS, "jax_params_diag.json"), "w"), indent=2)
    sys.exit(1)
P = globals()["P"]


def describe():
    leaves = []
    types = {}
    for mod, sub in P.items():
        for k, v in sub.items():
            key = f"{type(v).__name__}|{type(v.dtype).__name__}|{v.dtype!s}|{getattr(v.dtype, 'type', None).__module__ if hasattr(v.dtype, 'type') else '?'}"
            types[key] = types.get(key, 0) + 1
            leaves.append((mod, k, v.shape, str(v.dtype)))
    return {"n_modules": len(P), "n_leaves": len(leaves), "type_counts": types,
            "first_keys": [f"{m}/{k} {s} {d}" for m, k, s, d in leaves[:5]],
            "bf16_type_is_ml_dtypes": any(getattr(v.dtype, "type", None) is ml_dtypes.bfloat16 for s in P.values() for v in s.values()),
            "bf16_type_ids": sorted({id(v.dtype.type) for s in P.values() for v in s.values() if str(v.dtype) == "bfloat16"}),
            "ml_dtypes_bf16_id": id(ml_dtypes.bfloat16), "jnp_bf16_id": id(jnp.bfloat16),
            "np_dtype_hash_after": hash(np.dtype(ml_dtypes.bfloat16)),
            "sys_modules_bf16": sorted(m for m in sys.modules if "ml_dtypes" in m or "xla_extension" in m)[:20]}


step("describe_params", describe)
step("PRNGKey_after_load", lambda: str(jax.random.PRNGKey(0)))
step("jnp_promote_after_load", lambda: str(jnp.result_type(jnp.int32, jnp.uint32)))


def save_clean():
    d = os.path.join(ROOT, "xdg", "params_clean")
    os.makedirs(d, exist_ok=True)
    man = []
    for i, (mod, sub) in enumerate(P.items()):
        for k, v in sub.items():
            a = np.asarray(v)
            dt = str(a.dtype)
            if dt == "bfloat16":
                a = a.view(np.uint16)
            fn = f"leaf_{len(man):05d}.npy"
            np.save(os.path.join(d, fn), a, allow_pickle=False)
            man.append({"module": mod, "name": k, "file": fn, "dtype": dt, "shape": list(a.shape),
                        "sha256": hashlib.sha256(a.tobytes()).hexdigest()})
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"))
    return {"dir": d, "n_leaves": len(man), "gb": sum(os.path.getsize(os.path.join(d, m["file"])) for m in man) / 1e9}


step("save_clean_params", save_clean)
json.dump(out, open(os.path.join(RESULTS, "jax_params_diag.json"), "w"), indent=2)
print("wrote", os.path.join(RESULTS, "jax_params_diag.json"))
