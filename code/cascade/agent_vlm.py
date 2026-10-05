"""Step 2 (server GPU, after training): the cloud agent. A vision-language model verifies every frame that
the edge detector could escalate (s >= --s-min) and returns g = P("Yes") for "is there real smoke/fire".

    python -m cascade.agent_vlm --dataset ~/PROJECT/lha-yolo26/datasets/FireSmoke-Clean_v0.2 \
        --det ~/PROJECT/runs/cascade/det_yolo26-base_s0.csv \
        --out ~/PROJECT/runs/cascade/agent_qwen3b_yolo26-base_s0.csv --model Qwen/Qwen2.5-VL-3B-Instruct

g is read from the logits of the first answer token (no sampling). It is a RANKING score, not a calibrated
probability: g = sigmoid(logit_Yes - logit_No); both log-probabilities are stored. The configuration
(model revision, prompt hash, view, dtype, versions) is written to <out>.meta.json. The detector's top boxes are drawn on the frame (view=overlay) or the
frame is cropped around the top box (view=crop). Resumable: frames already in --out are skipped.

--dry replaces the VLM by a colour heuristic. It only exists to test the pipeline without a GPU:
never report --dry numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cascade.dataio import EVAL_SPLITS, ResumableCSV, read_csv_by_uid, read_manifest  # noqa: E402

FIELDS = ["uid", "g", "logit_yes", "logit_no", "agent_ms", "n_tokens", "agent", "view", "nbytes_payload",
          "pair_used"]

PROMPT = ("You are verifying an automatic fire alarm from a camera. An edge detector marked the red box(es) "
          "as possible smoke or fire; the boxes may be wrong. Look at the whole image. Is there real smoke "
          "from a fire or visible flames in this image? Clouds, fog, haze, dust, steam, sun glare, lamps and "
          "reflections do not count. Answer Yes or No.")
PROMPT_PLAIN = ("You are verifying an automatic fire alarm from a camera. Is there real smoke from a fire or "
                "visible flames in this image? Clouds, fog, haze, dust, steam, sun glare, lamps and reflections "
                "do not count. Answer Yes or No.")
PROMPT_CROP = ("You are verifying an automatic fire alarm from a camera. This is a crop around the region an "
               "edge detector flagged. Is there real smoke from a fire or visible flames? Clouds, fog, haze, "
               "dust, steam, sun glare, lamps and reflections do not count. Answer Yes or No.")


def previous_frame(rows, gap_min=30.0):
    """uid -> path of the previous frame of the same camera sequence (time-stamped sources only)."""
    from cascade.dataio import pyro_camera_time
    seq = {}
    for r in rows:
        stem = Path(r["image"]).stem.split("__", 1)[-1]
        ct = pyro_camera_time(stem)
        if ct is not None:
            seq.setdefault((r["split"], ct[0]), []).append((ct[1], r["uid"], r["path"]))
    out = {}
    for lst in seq.values():
        lst.sort()
        for (t0, _, p0), (t1, u1, _) in zip(lst, lst[1:]):
            if (t1 - t0).total_seconds() <= gap_min * 60:
                out[u1] = p0
    return out


def parse_boxes(s):
    out = []
    for part in filter(None, (s or "").split(";")):
        c, conf, *xy = part.split(":")
        out.append((int(c), float(conf), [float(v) for v in xy]))
    return out


def box_meta(boxes):
    """The box metadata the cloud needs for the overlay view (drawn there, not at the edge)."""
    return ";".join(f"{c}:{conf:.3f}:" + ":".join(f"{v:.4f}" for v in xy) for c, conf, xy in boxes)


def payload_image(path, boxes, view, max_side, quality, min_conf=0.05, pad=1.5, zoom=1.0, prev_path=None):
    """ONE measurement path: prepare -> encode -> count bytes -> decode -> the VLM sees exactly those bytes.

    quality = 0 disables compression (legacy path: the VLM sees the uncompressed image, payload = nan, so no
    communication claim can be attached to those scores).
    Payload per view (application payload only; no network headers / retransmissions):
      crop    : JPEG(crop at max_side)                          -> cloud needs no boxes
      overlay : JPEG(frame at max_side) + box metadata bytes    -> boxes are drawn in the cloud after decoding
      plain   : JPEG(frame at max_side)
    """
    import io

    from PIL import Image, ImageDraw
    im = Image.open(path).convert("RGB")
    w, h = im.size
    boxes = [b for b in boxes if b[1] >= min_conf] or boxes[:1]
    if view == "crop" and boxes:
        x1, y1, x2, y2 = boxes[0][2]
        cx, cy = (x1 + x2) / 2 * w, (y1 + y2) / 2 * h
        bw, bh = max((x2 - x1) * w, 64) * pad, max((y2 - y1) * h, 64) * pad
        box = (int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2)),
               int(min(w, cx + bw / 2)), int(min(h, cy + bh / 2)))
        im = im.crop(box)
        if prev_path is not None:                       # temporal pair: same box, previous frame on the left
            pv = Image.open(prev_path).convert("RGB").crop(box)
            comp = Image.new("RGB", (im.width * 2 + 8, max(im.height, pv.height)), (255, 255, 255))
            comp.paste(pv.resize(im.size, Image.BILINEAR), (0, 0))
            comp.paste(im, (im.width + 8, 0))
            im = comp
    target = max_side * float(zoom)
    k = target / max(im.size)
    if k < 1 or zoom > 1.0:
        im = im.resize((max(28, round(im.width * k)), max(28, round(im.height * k))), Image.BILINEAR)
    nbytes = float("nan")
    if quality:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=int(quality))
        raw = buf.getvalue()
        nbytes = float(len(raw))
        im = Image.open(io.BytesIO(raw)).convert("RGB")          # the model sees the decoded payload
        if view == "overlay":
            nbytes += float(len(box_meta(boxes).encode()))
    if view == "overlay":
        d = ImageDraw.Draw(im)
        lw = max(2, round(max(im.size) / 250))
        for _, _, (x1, y1, x2, y2) in boxes:
            d.rectangle([x1 * im.width, y1 * im.height, x2 * im.width, y2 * im.height], outline=(255, 0, 0),
                        width=lw)
    return im, nbytes


def prepare_image(path, boxes, view, max_side, min_conf=0.05):
    from PIL import Image, ImageDraw
    im = Image.open(path).convert("RGB")
    w, h = im.size
    boxes = [b for b in boxes if b[1] >= min_conf] or boxes[:1]
    if view == "crop" and boxes:
        x1, y1, x2, y2 = boxes[0][2]
        cx, cy = (x1 + x2) / 2 * w, (y1 + y2) / 2 * h
        bw, bh = max((x2 - x1) * w, 64) * 1.5, max((y2 - y1) * h, 64) * 1.5
        im = im.crop((int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2)),
                      int(min(w, cx + bw / 2)), int(min(h, cy + bh / 2))))
    elif view == "overlay":
        d = ImageDraw.Draw(im)
        lw = max(2, round(max(w, h) / 250))
        for _, _, (x1, y1, x2, y2) in boxes:
            d.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline=(255, 0, 0), width=lw)
    k = max_side / max(im.size)
    if k < 1:
        im = im.resize((max(28, round(im.width * k)), max(28, round(im.height * k))), Image.BILINEAR)
    return im


class VLMAgent:
    def __init__(self, model_id, device="cuda:0", quant="none", max_pixels=640 * 640):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:  # transformers < 4.50
            from transformers import Qwen2_5_VLForConditionalGeneration as AutoVLM
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id, min_pixels=64 * 28 * 28, max_pixels=max_pixels)
        self.processor.tokenizer.padding_side = "left"
        kw = {"device_map": {"": device}}
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        if not str(device).startswith("cuda"):
            dtype = torch.float32
        if quant == "4bit":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=dtype)
        try:
            self.model = AutoVLM.from_pretrained(model_id, dtype=dtype, **kw)
        except TypeError:  # transformers 4.x
            self.model = AutoVLM.from_pretrained(model_id, torch_dtype=dtype, **kw)
        self.model.eval()
        self.device = device
        self.dtype = str(dtype)
        self.revision = getattr(self.model.config, "_commit_hash", None) or ""
        tok = self.processor.tokenizer
        self.yes = self._ids(tok, ["Yes", "yes", " Yes", " yes", "YES"])
        self.no = self._ids(tok, ["No", "no", " No", " no", "NO"])
        if not self.yes or not self.no:
            raise SystemExit("could not find single-token Yes/No ids in this tokenizer")

    @staticmethod
    def _ids(tok, words):
        ids = set()
        for w in words:
            t = tok.encode(w, add_special_tokens=False)
            if len(t) == 1:
                ids.add(t[0])
        return sorted(ids)

    def score(self, images, prompt):
        torch = self.torch
        msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
                for _ in images]
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        inp = self.processor(text=texts, images=images, padding=True, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            logits = self.model(**inp).logits[:, -1, :].float()
        lp = torch.log_softmax(logits, -1)
        ly = torch.logsumexp(lp[:, self.yes], -1)
        ln = torch.logsumexp(lp[:, self.no], -1)
        g = torch.sigmoid(ly - ln)
        ntok = inp["attention_mask"].sum(1)
        return g.cpu().numpy(), ly.cpu().numpy(), ln.cpu().numpy(), ntok.cpu().numpy()


class DryAgent:
    """Colour heuristic (NOT a result): share of fire-coloured / grey-smoke pixels near the boxes."""

    def score(self, images, prompt):
        g = []
        for im in images:
            a = np.asarray(im, dtype=np.float32) / 255
            r, gg, b = a[..., 0], a[..., 1], a[..., 2]
            fire = ((r > 0.6) & (r > gg * 1.2) & (gg > b)).mean()
            smoke = ((np.abs(r - gg) < 0.05) & (np.abs(gg - b) < 0.05) & (r > 0.4) & (r < 0.85)).mean()
            z = 12 * fire + 3 * smoke - 1.5
            g.append(1 / (1 + math.exp(-z)))
        g = np.array(g)
        n = np.full(len(g), -1)
        return g, np.log(g + 1e-9), np.log(1 - g + 1e-9), n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--det", required=True, help="CSV from cascade.dump_detector")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--quant", choices=["none", "4bit"], default="none", help="4bit for the 7B model on 16 GB")
    ap.add_argument("--view", choices=["overlay", "crop", "plain"], default="overlay")
    ap.add_argument("--splits", nargs="+", default=list(EVAL_SPLITS))
    ap.add_argument("--s-min", type=float, default=0.02, help="frames below this edge score are never escalated")
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--crop-pad", type=float, default=1.5, help="crop context multiplier (1.5 = default box)")
    ap.add_argument("--zoom", type=float, default=1.0, help=">1 upscales the view (long side = max_side * zoom)")
    ap.add_argument("--pair", choices=["none", "prev"], default="none",
                    help="prev: put the previous frame of the same camera sequence beside the crop "
                         "(time-stamped sources only; pair_used = 0 where no previous frame exists)")
    ap.add_argument("--jpeg-quality", type=int, default=0,
                    help="0 = legacy (uncompressed input, payload NOT measured); >0 = encode/decode the payload "
                         "the cloud actually receives and record its bytes (required for any payload claim)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry", action="store_true", help="colour heuristic instead of the VLM (pipeline test only)")
    a = ap.parse_args()

    det = read_csv_by_uid(a.det)
    all_rows = read_manifest(a.dataset, a.splits)
    prev_of = previous_frame(all_rows) if a.pair == "prev" else {}
    rows = [r for r in all_rows if r["uid"] in det and float(det[r["uid"]]["s"]) >= a.s_min]
    if a.limit:
        from cascade.dataio import spread
        rows = spread(rows, a.limit)
    out = ResumableCSV(a.out, FIELDS)
    todo = [r for r in rows if r["uid"] not in out.done]
    print(f"{len(rows)} frames with s >= {a.s_min}, {len(todo)} to run", flush=True)
    if not todo:
        return
    agent = DryAgent() if a.dry else VLMAgent(a.model, a.device, a.quant)
    name = "dry-heuristic" if a.dry else Path(a.model).name + ("-4bit" if a.quant == "4bit" else "")
    prompt = {"overlay": PROMPT, "crop": PROMPT_CROP, "plain": PROMPT_PLAIN}[a.view]
    det_meta = Path(a.det).expanduser().with_suffix(".meta.json")
    meta = {"agent": name, "model": a.model, "revision": getattr(agent, "revision", ""),
            "dtype": getattr(agent, "dtype", ""), "quant": a.quant, "view": a.view, "max_side": a.max_side,
            "s_min": a.s_min, "jpeg_quality": a.jpeg_quality, "crop_pad": a.crop_pad, "zoom": a.zoom,
            "pair": a.pair,
            "payload_spec": ("prepare -> JPEG(q) -> bytes -> decode -> model; overlay boxes drawn after decode, "
                             "payload = image bytes + box metadata bytes (application payload only)")
            if a.jpeg_quality else "legacy: uncompressed input, payload not measured",
            "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "score": "g = sigmoid(logsumexp(logit_Yes-ids) - logsumexp(logit_No-ids)) at the first answer token",
            "yes_ids": getattr(agent, "yes", None), "no_ids": getattr(agent, "no", None),
            "det_meta_sha256": hashlib.sha256(det_meta.read_bytes()).hexdigest() if det_meta.exists() else "",
            "det_csv_sha256": hashlib.sha256(Path(a.det).expanduser().read_bytes()).hexdigest(),
            "dry": a.dry}
    try:
        import torch
        import transformers
        meta.update(torch=torch.__version__, transformers=transformers.__version__)
    except ImportError:
        pass
    meta_path = Path(a.out).expanduser().with_suffix(".meta.json")
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        diff = {k: (old.get(k), v) for k, v in meta.items() if old.get(k) != v}
        if diff:
            raise SystemExit(f"{a.out} was produced with a different configuration: {list(diff)}. Use another --out.")
    meta_path.write_text(json.dumps(meta, indent=1))
    t0, n = time.time(), 0
    try:
        for i in range(0, len(todo), a.batch):
            part = todo[i:i + a.batch]
            prepared = [payload_image(r["path"], parse_boxes(det[r["uid"]]["boxes"]), a.view, a.max_side,
                                      a.jpeg_quality, pad=a.crop_pad, zoom=a.zoom,
                                      prev_path=prev_of.get(r["uid"])) for r in part]
            ims = [x[0] for x in prepared]
            nb = [x[1] for x in prepared]
            t1 = time.perf_counter()
            g, ly, ln, nt = agent.score(ims, prompt)
            ms = (time.perf_counter() - t1) * 1000 / len(part)
            out.write([{"uid": r["uid"], "g": f"{g[j]:.5f}", "logit_yes": f"{ly[j]:.4f}", "logit_no": f"{ln[j]:.4f}",
                        "agent_ms": f"{ms:.1f}", "n_tokens": int(nt[j]), "agent": name, "view": a.view,
                        "nbytes_payload": "" if not np.isfinite(nb[j]) else f"{nb[j]:.0f}",
                        "pair_used": int(prev_of.get(r["uid"]) is not None)}
                       for j, r in enumerate(part)])
            n += len(part)
            if n % (a.batch * 50) < a.batch or n == len(todo):
                el = time.time() - t0
                print(f"  {n}/{len(todo)}  {n / el:.2f} img/s  ETA {(len(todo) - n) / (n / el) / 60:.1f} min", flush=True)
    finally:
        out.close()
    print("done ->", a.out)


if __name__ == "__main__":
    main()
