"""Phase B of the certificate-predicts-editability study (pre-registered plan:
docs/certificate-predicts-editability-plan-2026-07-02.md).

Runs knowledge editors from the locally patched EasyEdit clone on CounterFact
records with gpt2-xl, single-edit-from-fresh-model protocol:
  edit -> evaluate (efficacy / paraphrase / neighborhood, argmax + prob) ->
  restore original weights -> verify restoration with a fixed probe forward.

Record slice: the first STUDY_N (300) of the first 1,000 records (file order)
whose target_new first BPE token (with leading space) is well-defined - the
same deterministic ordering Phase A uses.

Restoration: a master snapshot of the editor's editable weights is taken at
process start and copied back after EVERY record (robust to partial editor
failures); GRACE is restored by swapping the original module object back.
Restoration is verified after every record via max|delta logits| on a fixed
probe prompt; the run aborts if it ever exceeds 1e-3.

Import bypass identical to the known-good run_rome_easyedit.py: torchaudio
neutralized before transformers, permissive stubs for unused LLM-API clients
(stubs satisfy easyeditor's optional LLM-API imports; harmless when the real
packages are absent), multimodal trainer imports already commented out in the
local clone.

EasyEdit: clone github.com/zjunlp/EasyEdit into scripts/EasyEdit; see
scripts/EASYEDIT_LINUX_JOB.md.

Usage:
  python scripts/run_editors_counterfact.py --editor ROME [--n-cases 3] [--out PATH]
Editors: BASE (no edit; pre-edit reference numbers), ROME, GRACE, FT, MEMIT,
AlphaEdit. Results append to the shared JSONL; (editor, case_id) pairs already
present are skipped, so smoke tests and resumed runs compose.
"""
import sys, types, os, json, time, argparse, random

try:
    import torchaudio  # noqa: F401
except Exception:
    sys.modules["torchaudio"] = None  # keep transformers from importing a broken torchaudio


def _stub(n):
    m = types.ModuleType(n)
    m.__getattr__ = lambda k: (_ for _ in ()).throw(AttributeError(k)) if (k.startswith("__") and k.endswith("__")) else type(k, (object,), {})
    sys.modules[n] = m


for _n in ["zhipuai", "dashscope", "vllm", "anthropic", "rouge"]:
    _stub(_n)

BASE = os.path.dirname(os.path.abspath(__file__))
EE = os.path.join(BASE, "EasyEdit")
sys.path.insert(0, EE)

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from hlm5.io import RESULTS_DIR

MODEL_ID = "gpt2-xl"
DEV = "cuda"
DATA = os.path.join(BASE, "data", "counterfact.json")
OUT_DEFAULT = str(RESULTS_DIR / "editors_counterfact_gpt2xl.jsonl")
STATS_DIR = os.path.join(EE, "data", "stats")
PROBE = "The Eiffel Tower is located in the city of"
POOL_N = 1000   # Phase A pool
STUDY_N = 300   # Phase B slice


def log(msg):
    print(msg, flush=True)


def load_model_with_retry():
    """Another agent may briefly share the GPU at start; on OOM wait 120s and retry."""
    for attempt in range(4):
        try:
            model = GPT2LMHeadModel.from_pretrained(MODEL_ID).to(DEV)
            model.eval()
            return model
        except torch.cuda.OutOfMemoryError:
            log(f"[load] CUDA OOM (attempt {attempt+1}); waiting 120s for GPU to free up")
            torch.cuda.empty_cache()
            time.sleep(120)
    raise RuntimeError("could not load model after 4 attempts")


def select_records(tok):
    if not os.path.exists(DATA):
        raise FileNotFoundError(
            f"{DATA} not found. Run `python scripts/fetch_counterfact.py` first "
            "to download CounterFact.")
    data = json.load(open(DATA, encoding="utf-8"))
    pool = []
    for rec in data:
        tn = rec["requested_rewrite"]["target_new"]["str"]
        if len(tok.encode(" " + tn)) >= 1:
            pool.append(rec)
        if len(pool) == POOL_N:
            break
    return pool[:STUDY_N]


def get_parameter(model, name):
    for n, p in model.named_parameters():
        if n == name:
            return p
    raise LookupError(name)


def setup_editor(editor, model, tok):
    """Returns (edit_fn, restore_fn, pre_eval_fn, note).
    edit_fn(rec) performs the edit in place. restore_fn() puts the model back.
    pre_eval_fn() is called before every eval forward (GRACE key_id reset)."""
    noop = lambda: None
    if editor == "BASE":
        return (lambda rec: None), noop, noop, "no edit; pre-edit reference metrics"

    if editor == "ROME":
        from easyeditor.models.rome.rome_main import apply_rome_to_model
        from easyeditor.models.rome.rome_hparams import ROMEHyperParams
        hp = ROMEHyperParams.from_hparams(os.path.join(EE, "hparams", "ROME", "gpt2-xl.yaml"))
        hp.model_name = MODEL_ID
        hp.device = 0
        hp.stats_dir = STATS_DIR
        names = [f"{hp.rewrite_module_tmp.format(l)}.weight" for l in hp.layers]
        snap = {n: get_parameter(model, n).detach().clone() for n in names}

        def edit(rec):
            rw = rec["requested_rewrite"]
            req = [{"prompt": rw["prompt"], "subject": rw["subject"],
                    "target_new": rw["target_new"]["str"]}]
            apply_rome_to_model(model, tok, req, hp, return_orig_weights=False)

        def restore():
            with torch.no_grad():
                for n, w in snap.items():
                    get_parameter(model, n).copy_(w)
        note = f"layers={hp.layers} mom2_adjustment={hp.mom2_adjustment}"
        return edit, restore, noop, note

    if editor == "MEMIT" or editor == "AlphaEdit":
        if editor == "MEMIT":
            from easyeditor.models.memit.memit_main import apply_memit_to_model as apply_fn
            from easyeditor.models.memit.memit_hparams import MEMITHyperParams as HP
            hp = HP.from_hparams(os.path.join(EE, "hparams", "MEMIT", "gpt2-xl.yaml"))
        else:
            from easyeditor.models.alphaedit.AlphaEdit_main import apply_AlphaEdit_to_model
            from easyeditor.models.alphaedit.AlphaEdit_hparams import AlphaEditHyperParams as HP
            hp = HP.from_hparams(os.path.join(EE, "hparams", "AlphaEdit", "gpt2-xl.yaml"))
            hp.P_loc = os.path.join(EE, "null_space_project_gpt2xl.pt")
            apply_fn = apply_AlphaEdit_to_model
        hp.model_name = MODEL_ID
        hp.device = 0
        hp.stats_dir = STATS_DIR
        names = [f"{hp.rewrite_module_tmp.format(l)}.weight" for l in hp.layers]
        snap = {n: get_parameter(model, n).detach().clone() for n in names}

        def edit(rec):
            rw = rec["requested_rewrite"]
            req = [{"prompt": rw["prompt"], "subject": rw["subject"],
                    "target_new": rw["target_new"]["str"]}]
            apply_fn(model, tok, req, hp, return_orig_weights=False)

        def restore():
            with torch.no_grad():
                for n, w in snap.items():
                    get_parameter(model, n).copy_(w)
        note = f"layers={hp.layers} mom2_adjustment={hp.mom2_adjustment}"
        return edit, restore, noop, note

    if editor == "FT":
        from easyeditor.models.ft.ft_main import apply_ft_to_model
        from easyeditor.models.ft.ft_hparams import FTHyperParams
        hp = FTHyperParams.from_hparams(os.path.join(EE, "hparams", "FT", "gpt2-xl.yaml"))
        hp.model_name = MODEL_ID
        hp.device = 0  # yaml says 4; this box has one GPU
        # FT touches every param whose name contains the module template (weight AND bias)
        names = [n for n, _ in model.named_parameters()
                 for l in hp.layers if hp.rewrite_module_tmp.format(l) in n]
        snap = {n: get_parameter(model, n).detach().clone() for n in names}

        def edit(rec):
            rw = rec["requested_rewrite"]
            prompt = rw["prompt"].format(rw["subject"])
            req = [{"prompt": prompt, "subject": rw["subject"],
                    "target_new": rw["target_new"]["str"]}]
            apply_ft_to_model(model, tok, req, hp, return_orig_weights=False)

        def restore():
            with torch.no_grad():
                for n, w in snap.items():
                    get_parameter(model, n).copy_(w)
        note = f"layers={hp.layers} objective={hp.objective_optimization} (FT-L) params={names}"
        return edit, restore, noop, note

    if editor == "GRACE":
        from easyeditor.models.grace.grace_main import apply_grace_to_model
        from easyeditor.models.grace.grace_hparams import GraceHyperParams
        hp = GraceHyperParams.from_hparams(os.path.join(EE, "hparams", "GRACE", "gpt2-xl.yaml"))
        hp.model_name = MODEL_ID
        hp.device = 0
        # inner_params: transformer.h[35].mlp.c_fc.weight -> module transformer.h.35.mlp.c_fc
        layer_path = hp.inner_params[0].rsplit(".", 1)[0].replace("[", ".").replace("]", "")
        parts = layer_path.split(".")
        parent = model
        for c in parts[:-1]:
            parent = parent[int(c)] if c.isdigit() else getattr(parent, c)
        attr = parts[-1]
        orig_layer = getattr(parent, attr)

        def edit(rec):
            rw = rec["requested_rewrite"]
            prompt = rw["prompt"].format(rw["subject"])
            req = [{"prompt": prompt, "target_new": rw["target_new"]["str"]}]
            apply_grace_to_model(model, tok, req, hp)

        def restore():
            setattr(parent, attr, orig_layer)  # adapter never mutates the wrapped layer

        def pre_eval():
            cur = getattr(parent, attr)
            if hasattr(cur, "key_id"):
                cur.key_id = -1  # adapter mutates key_id at inference; reset per forward
        note = f"inner_params={hp.inner_params} n_iter={hp.n_iter} eps={hp.eps}"
        return edit, restore, pre_eval, note

    raise ValueError(editor)


def main():
    random.seed(0)
    torch.manual_seed(0)

    ap = argparse.ArgumentParser()
    ap.add_argument("--editor", required=True,
                    choices=["BASE", "ROME", "GRACE", "FT", "MEMIT", "AlphaEdit"])
    ap.add_argument("--n-cases", type=int, default=STUDY_N,
                    help="process at most this many of the 300 study records")
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    records = select_records(tok)[: args.n_cases]
    log(f"[setup] {len(records)} study records; case_ids {records[0]['case_id']}..{records[-1]['case_id']}")

    model = load_model_with_retry()

    @torch.no_grad()
    def last_logits(text, pre_eval=lambda: None):
        pre_eval()
        ids = torch.tensor([tok.encode(text)], device=DEV)
        return model(input_ids=ids).logits[0, -1].float()

    edit_fn, restore_fn, pre_eval_fn, note = setup_editor(args.editor, model, tok)
    log(f"[setup] editor={args.editor} {note}")

    base_probe = last_logits(PROBE)

    def eval_record(rec, y_new, y_true):
        rw = rec["requested_rewrite"]
        out = {}

        def score(text):
            lg = last_logits(text, pre_eval_fn)
            lp = torch.log_softmax(lg, dim=-1)
            return int(lg.argmax()), lp[y_new].item(), lp[y_true].item()

        am, ln, lt = score(rw["prompt"].format(rw["subject"]))
        out["efficacy_argmax"] = int(am == y_new)
        out["efficacy_prob"] = int(ln > lt)
        out["prompt_logp_new"] = round(ln, 4)
        out["prompt_logp_true"] = round(lt, 4)

        pa, pp, pd = [], [], []
        for p in rec["paraphrase_prompts"]:
            am, ln, lt = score(p)
            pa.append(int(am == y_new)); pp.append(int(ln > lt)); pd.append(ln - lt)
        out["paraphrase_argmax"] = sum(pa) / len(pa)
        out["paraphrase_prob"] = sum(pp) / len(pp)
        out["paraphrase_logp_diff_mean"] = round(sum(pd) / len(pd), 4)
        out["n_paraphrase"] = len(pa)

        ns, na, nd = [], [], []
        for p in rec["neighborhood_prompts"]:
            am, ln, lt = score(p)
            ns.append(int(lt > ln)); na.append(int(am == y_new)); nd.append(lt - ln)
        out["neighborhood_score"] = sum(ns) / len(ns)          # target_true still beats target_new
        out["neighborhood_argmax_new"] = sum(na) / len(na)     # damage: argmax flipped to y_new
        out["neighborhood_logp_diff_mean"] = round(sum(nd) / len(nd), 4)
        out["n_neighborhood"] = len(ns)
        return out

    # resume support: skip (editor, case_id) pairs already written
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done.add((d["editor"], d["case_id"]))
                except Exception:
                    pass

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fout = open(args.out, "a", encoding="utf-8")
    n_done = n_err = 0
    t_edit_total = 0.0

    for i, rec in enumerate(records):
        cid = rec["case_id"]
        if (args.editor, cid) in done:
            continue
        rw = rec["requested_rewrite"]
        y_new = tok.encode(" " + rw["target_new"]["str"])[0]
        y_true = tok.encode(" " + rw["target_true"]["str"])[0]

        line = {"case_id": cid, "editor": args.editor, "y_new": y_new, "y_true": y_true}
        err = None
        metrics = None
        t_edit = None
        for attempt in range(2):
            try:
                t0 = time.time()
                edit_fn(rec)
                t_edit = time.time() - t0
                metrics = eval_record(rec, y_new, y_true)
                err = None
                break
            except torch.cuda.OutOfMemoryError as e:
                restore_fn()
                torch.cuda.empty_cache()
                err = f"CUDA OOM: {str(e)[:200]}"
                if i < 10 and attempt == 0:
                    log(f"[case {cid}] OOM early; waiting 120s then retrying once")
                    time.sleep(120)
                    continue
                break
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:300]}"
                break
        restore_fn()
        dev = (last_logits(PROBE) - base_probe).abs().max().item()

        if metrics:
            line.update(metrics)
        if t_edit is not None:
            line["edit_time_s"] = round(t_edit, 2)
            t_edit_total += t_edit
        if err:
            line["error"] = err
            n_err += 1
        line["restore_max_dev"] = round(dev, 6)
        fout.write(json.dumps(line) + "\n")
        fout.flush()
        n_done += 1

        if err:
            log(f"[{args.editor} {n_done}/{len(records)}] case {cid} ERROR {err}")
        else:
            log(f"[{args.editor} {n_done}/{len(records)}] case {cid} "
                f"eff_am={line['efficacy_argmax']} eff_pr={line['efficacy_prob']} "
                f"para_pr={line['paraphrase_prob']:.2f} neigh={line['neighborhood_score']:.2f} "
                f"t={line.get('edit_time_s', '?')}s dev={dev:.2e}")

        if dev > 1e-3:
            log(f"[FATAL] restoration failed on case {cid}: probe max dev {dev}")
            sys.exit(2)
        if i % 25 == 24:
            torch.cuda.empty_cache()

    fout.close()
    n_edits = max(n_done - n_err, 1)
    log(f"[done] {args.editor}: {n_done} records this run ({n_err} errors), "
        f"mean edit time {t_edit_total / n_edits:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
