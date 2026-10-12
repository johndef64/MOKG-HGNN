"""MOGNN-TF (gcn_tf) WITHOUT the shallow parallel branch.

The original model (benchmark/MOGNN-TF/multiomics_gnn/models/gcn_tf.py, class GCN)
already exposes the shallow branch through its `parallel` argument:

  parallel=True  -> parallel_linear1/2 are built, theta2 is computed from the projected
                    node features WITHOUT any graph convolution, and the classifier is
                    Linear(64 + 64, C) over [theta1 || theta2]                (:105-111, :335-343)
  parallel=False -> no parallel layers, no theta2, classifier Linear(64, C) over theta1 (:113)

So the variant is obtained by subclassing with parallel forced to False: nothing else
in the class changes (encoder, ChebConv layers, index-window pooling, linear1/2,
decoder on theta1, loss with reconstruction + L2). The training pipeline instantiates
GCN through builder.build_model from `model.parallel` in the config, which is exactly
what configs/mognntf_no_shallow.yaml sets; `verify_variant()` proves that the model built by the
pipeline is equivalent to this subclass and differs from the original only by the branch.

Run `python scripts/no_shallow/model_no_shallow.py` for the checks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402  (sets sys.path and cwd)

import torch  # noqa: E402
from multiomics_gnn.models.gcn_tf import GCN  # noqa: E402

SHALLOW_PARAM_PREFIXES = ("parallel_linear1.", "parallel_linear2.")


class GCNNoShallow(GCN):
    """gcn_tf with the shallow parallel branch removed (parallel forced to False)."""

    def __init__(self, *args, **kwargs):
        if "parallel" in kwargs:
            kwargs["parallel"] = False
        else:  # positional signature: (method, parallel, ...)
            args = list(args)
            args[1] = False
        super().__init__(*args, **kwargs)
        assert not self.parallel
        assert not hasattr(self, "parallel_linear1") and not hasattr(self, "parallel_linear2")
        assert self.classifier.in_features == self.concate_layer  # dim(theta1) only


def _build(cfg, num_classes=27):
    from multiomics_gnn.pancancer_prediction.training.builder import build_model
    torch.manual_seed(common.MODEL_SEED)
    model = build_model(cfg, num_classes, device="cpu")
    return model[0] if isinstance(model, tuple) else model


def verify_variant(level=700, seed=42):
    """Structural + functional checks. Returns a dict (all booleans must be True)."""
    base_with = common.load_yaml_config(common.ORIGINAL_CONFIG)
    base_without = common.load_yaml_config(common.NO_SHALLOW_CONFIG)
    panel = common.get_panel(level, seed, base_with)
    cfg_with = common.build_run_config(base_with, level, seed, panel, "unused")
    cfg_without = common.build_run_config(base_without, level, seed, panel, "unused")

    diff = common.config_diff(cfg_with, cfg_without)
    m_with, m_without = _build(cfg_with), _build(cfg_without)

    sh_with = {k: tuple(v.shape) for k, v in m_with.state_dict().items()}
    sh_without = {k: tuple(v.shape) for k, v in m_without.state_dict().items()}
    removed = sorted(set(sh_with) - set(sh_without))
    added = sorted(set(sh_without) - set(sh_with))
    reshaped = sorted(k for k in set(sh_with) & set(sh_without) if sh_with[k] != sh_without[k])

    # the pipeline-built variant must be equivalent to the explicit subclass
    sub = GCNNoShallow(**{k: getattr(m_without, a) for k, a in (
        ("method", "method"), ("parallel", "parallel"), ("l2", "l2"), ("decoder", "decoder"),
        ("poolsize", "poolsize"), ("poolrate", "poolrate"), ("edge_weights", "edge_weights"),
        ("edge_attributes", "edge_attributes"), ("num_gene", "num_gene"),
        ("num_mirna", "num_mirna"), ("num_tf", "num_tf"), ("omic_mode", "omic_mode"),
        ("num_classes", "num_classes"), ("jumping_knowledge", "jumping_knowledge"),
        ("jk_mode", "jk_mode"), ("dropout_rate", "dropout_rate"))})
    sh_sub = {k: tuple(v.shape) for k, v in sub.state_dict().items()}

    # functional: the classifier input of the variant is exactly theta1 (= linear2 output
    # after ReLU, then dropout which is the identity in eval mode)
    captured = {}
    m_without.eval()
    h1 = m_without.linear2.register_forward_hook(lambda m, i, o: captured.__setitem__("l2", o))
    h2 = m_without.classifier.register_forward_hook(
        lambda m, i, o: captured.__setitem__("cls_in", i[0]))
    n_nodes = m_without.total_nodes
    x = torch.randn(2, n_nodes, 2)
    idx = torch.arange(n_nodes)
    ei = torch.stack([torch.cat([idx, idx + n_nodes]), torch.cat([idx, idx + n_nodes])])
    with torch.no_grad():
        m_without(x, ei, torch.ones(ei.size(1)))
    h1.remove(); h2.remove()
    theta1_only = torch.allclose(captured["cls_in"], torch.relu(captured["l2"]))

    checks = {
        "config_differs_only_in_model.parallel": list(diff) == ["model.parallel"]
        and diff["model.parallel"] == (True, False),
        "removed_params_are_only_the_shallow_branch": bool(removed) and all(
            k.startswith(SHALLOW_PARAM_PREFIXES) for k in removed),
        "no_params_added": not added,
        "only_classifier_weight_reshaped": reshaped == ["classifier.weight"],
        "classifier_in_with": int(m_with.classifier.in_features),
        "classifier_in_without": int(m_without.classifier.in_features),
        "classifier_in_is_dim_theta1": m_without.classifier.in_features == m_without.concate_layer,
        "decoder_kept": hasattr(m_without, "decoder_1") and hasattr(m_without, "decoder_2"),
        "pipeline_model_equals_subclass": sh_sub == sh_without,
        "classifier_reads_theta1_only": bool(theta1_only),
        "removed_params": removed,
        "classifier_weight_shape": [sh_with["classifier.weight"], sh_without["classifier.weight"]],
    }
    checks["passed"] = all(v for k, v in checks.items() if isinstance(v, bool))
    return checks


if __name__ == "__main__":
    import json
    res = verify_variant()
    print(json.dumps(res, indent=2, default=str))
    sys.exit(0 if res["passed"] else 1)
