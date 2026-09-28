"""Cross-encoder step 2 (GPU): fine-tune intfloat/multilingual-e5-base (MIT, 278M params) as a pair classifier on
work/ce_train.parquet ("name | address" of the S2/S3 record vs the S1), then score work/ce_{val,test}.parquet
-> work/ce_pred_{val,test}.parquet (q, s1, p_ce).

  modal run tools/ce_gpu.py            # train + score
  modal run tools/ce_gpu.py --dry      # build the image, load the model, one tiny batch
"""
import modal

app = modal.App("amazon-ml-ce")
vol = modal.Volume.from_name("amazon-ml")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.4.1", "transformers==4.44.2", "polars==1.9.0", "pyarrow", "numpy<2", "sentencepiece")
MODEL, MAXLEN = "intfloat/multilingual-e5-base", 128  # run 023 (final); e5-small + bs 512, lr 8e-5 = run 022


@app.function(image=image, gpu="H100", volumes={"/vol": vol}, timeout=3600, cpu=8, memory=65536)
def run(dry: bool = False, max_train: int = 1_500_000, bs: int = 256, lr: float = 4e-5):
    import time

    import numpy as np
    import polars as pl
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    vol.reload()
    torch.backends.cuda.matmul.allow_tf32 = True
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).cuda()

    def enc(df):
        return tok(df["text_a"].to_list(), df["text_b"].to_list(), truncation=True, max_length=MAXLEN)["input_ids"]

    def batches(ids, order, size):
        for lo in range(0, len(order), size):
            idx = order[lo:lo + size]
            b = tok.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt")
            yield idx, {k: v.cuda(non_blocking=True) for k, v in b.items()}

    @torch.no_grad()
    def score(df, size=2048, chunk=500_000):
        """Chunked: tokenizing ~3M pairs in one call crashed the container (run 024)."""
        model.eval()
        out = []
        for lo in range(0, len(df), chunk):
            ids = enc(df[lo:lo + chunk])
            order = np.argsort([len(x) for x in ids])  # length-sorted: little padding
            p = np.zeros(len(ids), dtype=np.float32)
            for idx, b in batches(ids, order, size):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p[idx] = torch.sigmoid(model(**b).logits.float().squeeze(-1)).cpu().numpy()
            out.append(p)
        return np.concatenate(out)

    fake = pl.DataFrame({"q": ["a", "b"] * 2048, "s1": ["x", "y"] * 2048, "p": [0.9, 0.1] * 2048, "label": [1, 0] * 2048,
                         "text_a": ["Ram Traders | 12 MG Road, Pune", "Holdings Holdings | 82 Walnut Ave"] * 2048,
                         "text_b": ["Ram Traders Pvt Ltd | 12, M.G. Road, Pune", "Signature Holdings | 1 Walnut Avenue"] * 2048})
    tr = fake if dry else pl.read_parquet("/vol/work/ce_train.parquet")
    tr = tr.sample(min(max_train, len(tr)), seed=1, shuffle=True)
    t0 = time.time()
    ids, y = enc(tr), torch.tensor(tr["label"].to_numpy(), dtype=torch.float32)
    print(f"tokenized {len(ids)} train pairs in {time.time() - t0:.0f}s, pos {float(y.mean()):.3f}", flush=True)
    steps = (len(ids) + bs - 1) // bs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    order = np.random.default_rng(0).permutation(len(ids))
    t0 = time.time()
    for step, (idx, b) in enumerate(batches(ids, order, bs)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**b).logits.float().squeeze(-1)
        loss = lossf(logits, y[idx].cuda())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
        if step % 500 == 0:
            print(f"  step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    print(f"trained {steps} steps in {time.time() - t0:.0f}s", flush=True)
    for split in ("val", "test"):
        df = fake if dry else pl.read_parquet(f"/vol/work/ce_{split}.parquet")
        t0 = time.time()
        p = score(df)
        out = df.select("q", "s1").with_columns(p_ce=p)
        if "label" in df.columns:
            lab = df["label"].to_numpy()
            print(f"{split}: {len(df)} pairs in {time.time() - t0:.0f}s; mean p_ce pos {p[lab == 1].mean():.3f} "
                  f"neg {p[lab == 0].mean():.3f}; acc@0.5 {((p >= 0.5) == lab).mean():.4f}; "
                  f"lgb acc@0.5 {((df['p'].to_numpy() >= 0.5) == lab).mean():.4f}", flush=True)
        if not dry:
            out.write_parquet(f"/vol/work/ce_pred_{split}.parquet")
    if not dry:
        vol.commit()


@app.local_entrypoint()
def main(dry: bool = False, max_train: int = 1_500_000):
    run.remote(dry=dry, max_train=max_train)
