"""Reconstruct the exact prompt given to the language model for every scored claim.

Reviewer 1 of ACLing 2026 noted that the per-claim prompts were not retained. They were not,
but they are recoverable: the prompt is a deterministic function of the claim text and the
sentences the retriever returned at that budget, and both inputs are retained — the ranking
in retrieved_idx.npy and the sentence text in page_store.jsonl. Nothing in the construction
is stochastic.

So these prompts are RECONSTRUCTED from retained artifacts rather than recovered from a log,
and they are labelled that way. What they reproduce is the prompt, not the completion.

The digests of both inputs are recorded with the output so the reconstruction can be checked.
"""
from __future__ import annotations
import gzip, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidenceloop.corpus import load_claims, load_page_store
from evidenceloop import verify as V
from evidenceloop import llm_arm as LA

ROOT = Path(__file__).resolve().parents[2]
DATA, RES = ROOT / 'data_real', ROOT / 'results_real'
SUP = RES / 'supplement'
OUT = SUP / 'llm_arm_prompts_reconstructed.jsonl.gz'


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    claims = load_claims(DATA / 'fever_dev.jsonl')
    store = load_page_store(RES / 'page_store.jsonl')
    sent = V.sentence_frame(store)
    idx = np.load(RES / 'retrieved_idx.npy')
    texts = sent.sentence.to_numpy()
    assert idx.shape[0] == len(claims), 'the ranking must cover every claim'

    # the subsample the language model was scored on, reproduced the way llm_arm draws it
    rng = np.random.default_rng(LA.SEED)
    perm = rng.permutation(len(claims))
    test = perm[int(len(claims) * 0.5):]
    lab = claims.label.to_numpy()
    pick, per = [], LA.N_CLAIMS // len(LA.LABELS)
    for L in LA.LABELS:
        cand = test[lab[test] == L]
        pick.append(rng.choice(cand, size=min(per, len(cand)), replace=False))
    sel = np.sort(np.concatenate(pick))

    recorded = pd.read_csv(SUP / 'language_model_subsample_claim_ids.csv').claim_id.tolist()
    assert sorted(claims.claim_id.to_numpy()[sel].tolist()) == sorted(recorded), \
        'the reproduced subsample must match the identifiers retained with the study'

    n = 0
    with gzip.open(OUT, 'wt') as fh:
        for b in LA.BUDGETS:
            for i in sel:
                # exactly the selection llm_arm makes before calling the prompt builder
                s = [texts[j] for j in idx[i, :b] if 0 <= j < len(texts)]
                fh.write(json.dumps(dict(
                    claim_id=int(claims.claim_id.iloc[i]), budget=int(b),
                    label=str(lab[i]), n_sentences=len(s),
                    prompt=LA.user_message(claims.claim.iloc[i], s))) + '\n')
                n += 1
    expected = len(sel) * len(LA.BUDGETS)
    assert n == expected, f'{n} prompts written, expected {expected}'

    man = dict(
        status='COMPLETED',
        what='the prompt given to the language model for every scored claim at every budget',
        provenance='RECONSTRUCTED from retained artifacts, not recovered from a log. The '
                   'prompt is a deterministic function of the claim text and the retrieved '
                   'sentences, so it is reproduced exactly; the completions are not '
                   'reconstructible and are the re-run\'s.',
        prompts=n, claims=int(len(sel)), budgets=list(LA.BUDGETS),
        max_sentence_characters=LA.MAX_SENT_CHARS,
        inputs={
            'retrieved_idx.npy': sha(RES / 'retrieved_idx.npy'),
            'page_store.jsonl': sha(RES / 'page_store.jsonl'),
            'fever_dev.jsonl': sha(DATA / 'fever_dev.jsonl'),
        },
        builder='evidenceloop.llm_arm.user_message, the function the run executes')
    (SUP / 'llm_arm_prompts_manifest.json').write_text(json.dumps(man, indent=2))
    print(f'{OUT.name}: {n} prompts for {len(sel)} claims at {len(LA.BUDGETS)} budgets '
          f'({OUT.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
