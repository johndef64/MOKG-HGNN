"""Sanity check of the version-3 factorial experiment (plan §6). Writes the gate file
results/four_arms/sanity_check.json; run_four_arms.py refuses to start without passed=true.

  A  node vocabularies: all 35 (level, split) verified on their template (hetero_dirs.py)
  B  configs: the MOKG-HGNN run config equals the config.json of the existing kg runs
     (except output paths, hetero_dir and the arm label); mognntf_shallow_only.yaml differs
     from config_final.yml only by train.decoder
  C  kg_s model: same initial weights as kg except the new layers, shallow width 8N -> 2N
     -> 64, forward + backward on a real batch; sh_kg forward + backward
  D  sh_tf model: only pre_conv + shallow + classifier, shallow weights equal to mol_s
  R  (--repro N) arm kg re-run N times through this code at 700 genes, split 42: must match
     the existing run within 0.02 macro-F1 (checks hardware + per-run node vocabularies)

  python scripts/four_arms/sanity_check.py               # A-D (minutes, CPU is enough)
  python scripts/four_arms/sanity_check.py --repro 1     # + R (about 1 h on a GPU)
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402

NS = core.NS
TOL = 0.02
LEVEL, SEED = 700, 42
VOLATILE_MOKG = {"project.experiment_name", "paths.results_dir", "data.hetero_dir", "model.arm"}


def check_hetero_dirs():
    import hetero_dirs
    out = {}
    for g in core.LEVELS:
        for s in core.SEEDS:
            hetero_dirs.ensure(g, s)
            out[f"g{g}_s{s}"] = True
    return {"passed": all(out.values()), "pairs_verified": len(out)}


def check_configs():
    import yaml
    diffs, compared = {}, 0
    for (g, s), r in NS.read_multiscale_existing().items():
        old = json.loads((Path(r["run_dir"]) / "config.json").read_text())
        new = core.mokg_config("kg", g, s, core.HETERO / f"g{g}_s{s}", core.RUNS_RAW)
        fo, fn = NS.flatten(old), NS.flatten(new)
        d = {k: (fo.get(k), fn.get(k)) for k in set(fo) | set(fn)
             if k not in VOLATILE_MOKG and fo.get(k) != fn.get(k)}
        compared += 1
        if d:
            diffs[f"g{g}_s{s}"] = {k: list(v) for k, v in d.items()}
    tf_diff = NS.config_diff(NS.load_yaml_config(NS.ORIGINAL_CONFIG),
                             NS.load_yaml_config(core.SHALLOW_ONLY_TF_CONFIG))
    tf_ok = set(tf_diff) == {"train.decoder"}
    return {"passed": compared > 0 and not diffs and tf_ok, "kg_runs_compared": compared,
            "kg_config_differences": diffs, "sh_tf_config_diff": {k: list(map(str, v)) for k, v in tf_diff.items()}}


def check_mokg_models():
    import torch
    from torch_geometric.loader import DataLoader
    from multiomics_kg_hgnn.models.hetero_gnn import HeteroMultiScaleGNN
    from multiomics_kg_hgnn.pancancer_prediction.preprocessing.make_datasets import make_datasets
    import hetero_dirs
    import models

    cfg = core.mokg_config("kg_s", LEVEL, SEED, hetero_dirs.ensure(LEVEL, SEED), core.RUNS_RAW)
    d, m = cfg["data"], cfg["model"]
    train_ds, _, _, n_cls, _ = make_datasets(
        split_dir=d["split_dir"], template_path=d["template_path"], hetero_dir=d["hetero_dir"],
        use_cnv=d["use_cnv"], use_mirna=d["use_mirna"], scaler=d["scaler"])
    batch = next(iter(DataLoader(train_ds, batch_size=8, shuffle=False)))
    kw = dict(metadata=train_ds.template.metadata(), num_classes=n_cls,
              feature_dims={nt: t.shape[-1] for nt, t in train_ds.features.items()},
              backbone=m["backbone"], hidden=m["hidden"], num_layers=m["num_layers"],
              heads=m["heads"], dropout=m["dropout"], readout_types=tuple(m["readout_types"]))
    num_nodes = core.num_nodes_of(LEVEL, SEED)

    def built(factory):
        torch.manual_seed(core.MODEL_SEED)
        model = factory(**kw)
        model.eval()
        with torch.no_grad():
            model(batch)                       # materialises the lazy SAGEConv weights
        return model

    kg = built(models.mokg_factory("kg", num_nodes))
    kgs = built(models.mokg_factory("kg_s", num_nodes))
    a, b = kg.state_dict(), kgs.state_dict()
    new_layers = {"classifier.3.weight", "classifier.3.bias"}
    same_init = all(torch.equal(a[k], b[k]) for k in a if k not in new_layers)
    extra = sorted(set(b) - set(a))
    n_mol = sum(num_nodes[t] for t in ("gene", "miRNA", "TF"))
    sh = kgs.shallow
    widths = [sh.linear1.in_features, sh.linear1.out_features, sh.linear2.out_features]
    widths_ok = widths == [8 * n_mol, (8 * n_mol) // 4, 64] and \
        kgs.classifier[3].in_features == m["hidden"] + 64

    def trains(model):
        model.train()
        out = model(batch)
        torch.nn.functional.cross_entropy(out, batch.y).backward()
        # the new layers and the classifier must receive gradients (some last-layer
        # relations into miRNA/TF get none in the original model too: not read out)
        return tuple(out.shape) == (8, n_cls) and all(
            p.grad is not None for k, p in model.named_parameters()
            if k.startswith(("shallow.", "classifier.")))

    shk = built(models.mokg_factory("sh_kg", num_nodes))
    res = {"kg_factory_is_original": models.mokg_factory("kg", num_nodes) is HeteroMultiScaleGNN,
           "kg_s_same_initial_weights_as_kg": same_init,
           "kg_s_new_parameters": [k for k in extra if not k.startswith("shallow.")] == [],
           "kg_s_widths": widths, "kg_s_widths_ok": widths_ok,
           "kg_s_forward_backward": trains(kgs), "sh_kg_forward_backward": trains(shk),
           "n_params": {"kg": sum(p.numel() for p in kg.parameters()),
                        "kg_s": sum(p.numel() for p in kgs.parameters()),
                        "sh_kg": sum(p.numel() for p in shk.parameters())}}
    res["passed"] = all(v for k, v in res.items() if isinstance(v, bool))
    return res


def check_tf_model():
    import torch
    from multiomics_gnn.pancancer_prediction.training.builder import build_model

    def built(cfg_path, swap):
        base = NS.load_yaml_config(cfg_path)
        cfg = NS.build_run_config(base, LEVEL, SEED, NS.get_panel(LEVEL, SEED, base), "unused")
        torch.manual_seed(core.MODEL_SEED)
        if swap:
            with core.swapped_gcn():
                m = build_model(cfg, 27, device="cpu")
        else:
            m = build_model(cfg, 27, device="cpu")
        return (m[0] if isinstance(m, tuple) else m), cfg

    ws, _ = built(NS.ORIGINAL_CONFIG, False)
    sh, cfg = built(core.SHALLOW_ONLY_TF_CONFIG, True)
    a, b = ws.state_dict(), sh.state_dict()
    keys_ok = all(k.startswith(("pre_conv_linear", "parallel_linear", "classifier.")) for k in b)
    shared = [k for k in b if k.startswith(("pre_conv_linear", "parallel_linear"))]
    same = all(torch.equal(a[k], b[k]) for k in shared)
    n = cfg["data"]["num_gene"] + cfg["data"]["num_mirna"] + cfg["data"]["num_tf"]
    sh.train()
    out = sh(torch.randn(4, n, 2), None, None)
    out.sum().backward()
    res = {"type": type(sh).__name__, "only_shallow_parameters": keys_ok,
           "shallow_weights_equal_to_mol_s": same, "classifier_in": sh.classifier.in_features,
           "forward_shape_ok": tuple(out.shape) == (4, 27), "decoder_off": sh.decoder is False,
           "n_params": sum(p.numel() for p in sh.parameters())}
    res["passed"] = keys_ok and same and res["forward_shape_ok"] and res["decoder_off"] \
        and res["classifier_in"] == 64
    return res


def reproduction(repeats):
    ref = NS.read_multiscale_existing().get((LEVEL, SEED))
    if ref is None:
        return {"passed": False, "reason": f"no existing kg run for g{LEVEL} s{SEED} on this machine"}
    vals = []
    for i in range(repeats):
        rec = core.run_mokg("kg", LEVEL, SEED, raw_root=core.OUT / "sanity_runs")
        vals.append(rec["test_macro_f1"])
        print(f"[repro] {i + 1}/{repeats}: {rec['test_macro_f1']:.4f} (existing {ref['f1']:.4f})", flush=True)
    return {"passed": all(abs(v - ref["f1"]) <= TOL for v in vals), "existing": ref["f1"],
            "values": vals, "tolerance": TOL}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro", type=int, default=0, help="repetitions of the kg reproduction")
    a = ap.parse_args()
    out = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "env": NS.environment()}
    for name, fn in (("A_hetero_dirs", check_hetero_dirs), ("B_configs", check_configs),
                     ("C_mokg_models", check_mokg_models), ("D_tf_model", check_tf_model)):
        out[name] = fn()
        print(f"[sanity] {name}: {'OK' if out[name]['passed'] else 'FAILED'}", flush=True)
    if a.repro:
        out["R_reproduction"] = reproduction(a.repro)
        print(f"[sanity] R_reproduction: {'OK' if out['R_reproduction']['passed'] else 'FAILED'}")
    out["passed"] = all(v["passed"] for k, v in out.items() if isinstance(v, dict) and "passed" in v)
    out["reproduction_done"] = bool(a.repro)
    core.write_json(core.SANITY_JSON, out)
    print(json.dumps(out, indent=1))
    sys.exit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
