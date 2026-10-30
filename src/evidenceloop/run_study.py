from __future__ import annotations
import hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np, pandas as pd, sklearn, scipy
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidenceloop.corpus import load_claims, load_page_store
from evidenceloop import verify as V
ROOT = Path(__file__).resolve().parents[2]
DATA, OUT = ROOT/'data_real', ROOT/'results_real'; OUT.mkdir(exist_ok=True)
MAX_BUDGET = max(V.BUDGETS)
SPLIT = 0.5          # first half of the claim set trains the verifier, second half tests it
SEED = 0


def log(m):
    line = f'{time.strftime("%H:%M:%S")}  {m}'
    print(line, flush=True); open(OUT/'run.log', 'a').write(line + '\n')


def main():
    run_id = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()); log(f'RUN {run_id} start')
    claims = load_claims(DATA/'fever_dev.jsonl')
    log(f'claims: {len(claims):,}  ' +
        ', '.join(f'{k} {v:,}' for k, v in claims.label.value_counts().items()))
    store = load_page_store(OUT/'page_store.jsonl')
    meta = json.loads((OUT/'page_store_meta.json').read_text())
    sent = V.sentence_frame(store)
    log(f'retrieval pool: {len(store):,} pages, {len(sent):,} sentences '
        f'({meta["distractors"]:,} pages are sampled distractors)')

    v, X = V.build_index(sent)
    log(f'index built: {X.shape[0]:,} sentences by {X.shape[1]:,} terms')
    idx, score = V.retrieve(v, X, claims.claim.tolist(), MAX_BUDGET)
    log(f'retrieved top {MAX_BUDGET} sentences for every claim')
    np.save(OUT/'retrieved_idx.npy', idx[:, :MAX_BUDGET].astype(np.int32))

    rec = V.evidence_recall(claims, idx, sent)
    rec.to_csv(OUT/'evidence_recall.csv', index=False)
    log('evidence recall against sentences read:\n' + rec.round(4).to_string(index=False))

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, f1_score
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(claims))
    k = int(len(claims)*SPLIT)
    tr_i, te_i = perm[:k], perm[k:]
    rows, per_label = [], []
    for b in V.BUDGETS:
        f = V.features(claims, idx, score, sent, b)
        y = claims.label.to_numpy()
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=5,
                                           random_state=SEED).fit(f.iloc[tr_i][V.FEATURES],
                                                                  y[tr_i])
        p = m.predict(f.iloc[te_i][V.FEATURES])
        yt = y[te_i]
        rows.append(dict(budget=b, accuracy=float(accuracy_score(yt, p)),
                         macro_f1=float(f1_score(yt, p, average='macro'))))
        for lab in V.LABELS:
            mask = yt == lab
            rows[-1][f'recall_{lab}'] = float((p[mask] == lab).mean())
            pred = p == lab
            rows[-1][f'precision_{lab}'] = float((yt[pred] == lab).mean()) if pred.any() else np.nan
        per_label.append(pd.DataFrame(dict(budget=b, claim_id=claims.claim_id.to_numpy()[te_i],
                                           true=yt, pred=p)))
        log(f'  budget {b:>3}: accuracy {rows[-1]["accuracy"]:.4f}, '
            f'withhold recall {rows[-1]["recall_NOT ENOUGH INFO"]:.4f}, '
            f'support recall {rows[-1]["recall_SUPPORTS"]:.4f}')
    # ---- oracle retrieval: for a claim that can be settled, put its gold sentences at the
    # top of the list. This is not a system an analyst could run; it separates the part of the
    # verification gap that better retrieval could close from the part it could not.
    key = {(t, l): i for i, (t, l) in enumerate(zip(sent.title, sent.line))}
    oidx = idx.copy()
    moved = 0
    for r, g in enumerate(claims.gold_sentences):
        gi = [key[(t, l)] for t, l in g if (t, l) in key]
        if not gi:
            continue
        moved += 1
        rest = [j for j in idx[r].tolist() if j not in set(gi)]
        oidx[r] = np.array((gi + rest)[:MAX_BUDGET])
    log(f'oracle retrieval: gold sentences promoted for {moved:,} claims that can be settled')
    orows, oper = [], []
    for b in V.BUDGETS:
        f = V.features(claims, oidx, score, sent, b)
        y = claims.label.to_numpy()
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=5,
                                           random_state=SEED).fit(f.iloc[tr_i][V.FEATURES],
                                                                  y[tr_i])
        p = m.predict(f.iloc[te_i][V.FEATURES]); yt = y[te_i]
        oper.append(pd.DataFrame(dict(budget=b, claim_id=claims.claim_id.to_numpy()[te_i],
                                      true=yt, pred=p)))
        orows.append(dict(budget=b, accuracy=float(accuracy_score(yt, p)),
                          macro_f1=float(f1_score(yt, p, average='macro')),
                          **{f'recall_{lab}': float((p[yt == lab] == lab).mean())
                             for lab in V.LABELS}))
    ores = pd.DataFrame(orows); ores.to_csv(OUT/'verification_oracle_retrieval.csv',
                                            index=False)
    pd.concat(oper, ignore_index=True).to_csv(OUT/'predictions_oracle.csv.gz', index=False,
                                              compression='gzip')
    log('verification with oracle retrieval:\n' +
        ores[['budget', 'accuracy', 'macro_f1'] +
             [f'recall_{l}' for l in V.LABELS]].round(4).to_string(index=False))

    res = pd.DataFrame(rows); res.to_csv(OUT/'verification_by_budget.csv', index=False)
    pd.concat(per_label, ignore_index=True).to_csv(OUT/'predictions.csv.gz', index=False,
                                                   compression='gzip')
    log('verification against sentences read:\n' +
        res[['budget', 'accuracy', 'macro_f1'] +
            [f'recall_{l}' for l in V.LABELS]].round(4).to_string(index=False))

    man = dict(
        run_id=run_id, status='COMPLETED',
        data='FEVER: human-written claims against an encyclopaedia snapshot, labelled '
             'SUPPORTS, REFUTES or NOT ENOUGH INFO, with the sentences an annotator used',
        provenance=dict(
            claims_file='fever_dev.jsonl',
            claims_sha256=hashlib.sha256((DATA/'fever_dev.jsonl').read_bytes()).hexdigest(),
            snapshot='wiki-pages.zip',
            retrieval_pool=meta),
        llm_arm=dict(
            status='NOT EXECUTED',
            reason=('The project specification asks for an executed language model producing '
                    'evidence-linked claims with retained prompts and raw outputs. No language '
                    'model was executed in this study and none of its results depend on one. '
                    'The claims are the corpus authors\' and the verifier is a retrieval and '
                    'classifier pipeline. Nothing here should be read as a measurement of any '
                    'language model.')),
        oracle_retrieval=('For a claim that can be settled, the annotator\'s own sentences '
                          'are promoted to the top of the retrieved list. No analyst could do '
                          'this, so it is an upper bound that separates what better retrieval '
                          'could buy from what it could not.'),
        budgets=V.BUDGETS, features=V.FEATURES, split=SPLIT, seed=SEED,
        n_claims=int(len(claims)),
        label_counts={k: int(v) for k, v in claims.label.value_counts().items()},
        environment=dict(python=platform.python_version(), platform=platform.platform(),
                         numpy=np.__version__, pandas=pd.__version__,
                         sklearn=sklearn.__version__, scipy=scipy.__version__),
        limitations=[
            'The retrieval pool is every cited page plus a sampled 120,000 distractors rather '
            'than the full 5.4 million page snapshot. A smaller pool makes retrieval easier, '
            'so every cost reported here is a lower bound on the cost against the whole '
            'snapshot.',
            'Cost is counted in sentences read. It does not include the cost of producing the '
            'retrieval ranking itself, which a real verifier also pays.',
            'A NOT ENOUGH INFO claim has no gold evidence by construction, so evidence recall '
            'is defined only for the claims that can be settled. That is a property of the '
            'task rather than a choice made here.',
            'No language model was executed. The specification asks for one and its absence '
            'is recorded rather than worked around; nothing in these results is a measurement '
            'of a language model.',
            'The claims are written by annotators to be checkable against this snapshot. '
            'Claims arising in the wild would be messier and the share that cannot be settled '
            'would not be one third by construction.',
            'The verifier reads retrieved sentences through summary features rather than '
            'reading them as language. A reader that understood the sentences would do better '
            'on every budget, which would raise the curves without changing which label '
            'benefits from a larger budget.'])
    (OUT/'run_manifest.json').write_text(json.dumps(man, indent=2))
    log(f'run_manifest.json written; RUN {run_id} COMPLETED')


if __name__ == '__main__':
    main()
