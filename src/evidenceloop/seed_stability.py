"""Is the oracle arm's non-monotonicity a property of the data or of one random split?

Reviewer 3 of ACLing 2026 asked for the non-monotonicity to be explained. The paired
intervals show it is not separable from zero, which bounds it but does not say what it is.
This asks the question directly: hold everything fixed except the seed, which controls both
the train/test split and the classifier, and see whether the dip stays where it is.

If the dip lands at a different budget on each seed it is resampling variation and that is
the explanation. If it lands at the same budget every time it is systematic, the paper's
current wording is too weak and must change.

Features depend on the retrieval and the budget, not on the seed, so they are computed once
and reused. Each seed is written to disk as it finishes.

No duration of this analysis is reported anywhere.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidenceloop.corpus import load_claims, load_page_store
from evidenceloop import verify as V

ROOT = Path(__file__).resolve().parents[2]
DATA, RES = ROOT / 'data_real', ROOT / 'results_real'
SUP = RES / 'supplement'
OUT = SUP / 'seed_stability.csv'
SEEDS = list(range(20))
SPLIT = 0.5


def log(m):
    line = f'{time.strftime("%H:%M:%S")}  {m}'
    print(line, flush=True)
    (SUP / 'seed_stability.log').open('a').write(line + '\n')


def main():
    SUP.mkdir(parents=True, exist_ok=True)
    claims = load_claims(DATA / 'fever_dev.jsonl')
    store = load_page_store(RES / 'page_store.jsonl')
    sent = V.sentence_frame(store)
    idx = np.load(RES / 'retrieved_idx.npy')
    y = claims.label.to_numpy()
    log(f'claims {len(claims):,}  sentences {len(sent):,}')

    # the score matrix was never retained, so it is rebuilt here
    v, X = V.build_index(sent)
    _, score = V.retrieve(v, X, claims.claim.tolist(), max(V.BUDGETS))
    log('retrieval rebuilt')

    key = {(t, l): i for i, (t, l) in enumerate(zip(sent.title, sent.line))}
    oidx = idx.copy()
    for r, g in enumerate(claims.gold_sentences):
        gi = [key[(t, l)] for t, l in g if (t, l) in key]
        if not gi:
            continue
        rest = [j for j in idx[r].tolist() if j not in set(gi)]
        oidx[r] = np.array((gi + rest)[:max(V.BUDGETS)])

    # features do not depend on the seed
    feats = {}
    for arm, ix in (('feasible', idx), ('annotator-first', oidx)):
        for b in V.BUDGETS:
            feats[(arm, b)] = V.features(claims, ix, score, sent, b)[V.FEATURES]
    log(f'features cached for {len(feats)} arm-budget combinations')

    rows, first = [], not OUT.exists()
    for seed in SEEDS:
        perm = np.random.default_rng(seed).permutation(len(claims))
        k = int(len(claims) * SPLIT)
        tr, te = perm[:k], perm[k:]
        out = []
        for arm in ('feasible', 'annotator-first'):
            for b in V.BUDGETS:
                f = feats[(arm, b)]
                m = HistGradientBoostingClassifier(
                    max_iter=300, learning_rate=0.08, max_depth=5,
                    random_state=seed).fit(f.iloc[tr], y[tr])
                p = m.predict(f.iloc[te])
                out.append(dict(seed=seed, arm=arm, budget=b,
                                accuracy=float(accuracy_score(y[te], p)),
                                macro_f1=float(f1_score(y[te], p, average='macro'))))
        d = pd.DataFrame(out)
        d.to_csv(OUT, mode='a', header=first, index=False)
        first = False
        rows.extend(out)
        o = d[d.arm == 'annotator-first'].set_index('budget').accuracy
        log(f'seed {seed:>2}: annotator-first minimum at budget {int(o.idxmin()):>3} '
            f'({o.min():.4f}), maximum at {int(o.idxmax()):>3} ({o.max():.4f})')

    (SUP / 'seed_stability_manifest.json').write_text(json.dumps(dict(
        status='COMPLETED', seeds=SEEDS, budgets=V.BUDGETS, split=SPLIT,
        question='whether the annotator-first arm\'s non-monotonicity is stable across '
                 'resampling, the seed controlling both the split and the classifier',
        note='features are seed-independent and were computed once; nothing else is shared '
             'between seeds'), indent=2))
    log('SEED STABILITY COMPLETED')


if __name__ == '__main__':
    main()
