"""Same-claim comparisons, done as paired comparisons.

Every accuracy comparison this study makes is between two conditions scored on the SAME
held-out claims: one budget against another, and oracle retrieval against the retrieval a
system could actually run. Treating those as independent samples throws away the pairing and
widens the uncertainty for no reason; it also cannot say whether a difference is real.

This module recomputes each of those comparisons as a paired one: a bootstrap interval over
claims for the difference in accuracy, and McNemar's exact test on the discordant pairs. It
also builds the per-claim retrieval record that lets the coverage-rises-accuracy-falls
question be asked of the data rather than argued about.

Written for the ACLing 2026 review. It adds analyses to retained predictions; it changes no
reported value and re-derives the published accuracies as a check.

No duration of this analysis is reported anywhere.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidenceloop.corpus import load_claims, load_page_store
from evidenceloop import verify as V

ROOT = Path(__file__).resolve().parents[2]
DATA, OUT = ROOT / 'data_real', ROOT / 'results_real'
SUP = OUT / 'supplement'
SEED, BOOT = 0, 10_000


def log(m):
    print(m, flush=True)
    open(OUT / 'paired_stats.log', 'a').write(str(m) + '\n')


def correct_matrix(path: Path) -> pd.DataFrame:
    """claims on the rows, budgets on the columns, 1 where the label was right."""
    d = pd.read_csv(path)
    d['ok'] = (d.pred == d.true).astype(np.int8)
    m = d.pivot(index='claim_id', columns='budget', values='ok')
    return m.sort_index()


def paired_diff(a: np.ndarray, b: np.ndarray, rng) -> dict:
    """Accuracy of b minus accuracy of a on the same claims, with a paired interval."""
    d = b.astype(float) - a.astype(float)
    n = len(d)
    i = rng.integers(0, n, size=(BOOT, n))
    boot = d[i].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    b_only = int(((a == 0) & (b == 1)).sum())      # wrong then right
    c_only = int(((a == 1) & (b == 0)).sum())      # right then wrong
    disc = b_only + c_only
    p = float(binomtest(b_only, disc, 0.5).pvalue) if disc else 1.0
    return dict(n_claims=n, accuracy_a=float(a.mean()), accuracy_b=float(b.mean()),
                difference=float(d.mean()), ci_low=float(lo), ci_high=float(hi),
                ci_excludes_zero=bool(lo > 0 or hi < 0),
                mcnemar_b_wrong_to_right=b_only, mcnemar_c_right_to_wrong=c_only,
                mcnemar_discordant=disc, mcnemar_p=p)


def per_claim_evidence(claims: pd.DataFrame, idx: np.ndarray,
                       sent: pd.DataFrame) -> pd.DataFrame:
    """For every claim and budget: was any annotator sentence retrieved, and a complete set."""
    key = {(t, l): i for i, (t, l) in enumerate(zip(sent.title, sent.line))}
    flat = [set(key[(t, l)] for t, l in g if (t, l) in key)
            for g in claims.gold_sentences]
    groups = []
    for gs in claims.gold_groups:
        gg = []
        for g in gs:
            if all((t, l) in key for t, l in g):
                gg.append(set(key[(t, l)] for t, l in g))
        groups.append(gg)
    cid = claims.claim_id.to_numpy()
    rows = []
    for b in V.BUDGETS:
        for r in range(len(claims)):
            top = set(idx[r, :b].tolist())
            rows.append((b, cid[r], bool(flat[r]),
                         bool(top & flat[r]) if flat[r] else False,
                         any(g <= top for g in groups[r]) if groups[r] else False))
    return pd.DataFrame(rows, columns=['budget', 'claim_id', 'settleable',
                                       'sentence_hit', 'complete_set_hit'])


def main():
    SUP.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    base = correct_matrix(OUT / 'predictions.csv.gz')
    orac = correct_matrix(OUT / 'predictions_oracle.csv.gz')
    assert list(base.index) == list(orac.index), 'the two arms were scored on the same claims'
    log(f'held-out claims {len(base):,}, budgets {list(base.columns)}')

    # the published accuracies, re-derived from the per-claim records as a check
    pub = pd.read_csv(OUT / 'verification_by_budget.csv').set_index('budget')
    chk = []
    for b in base.columns:
        chk.append(dict(budget=b, from_per_claim=float(base[b].mean()),
                        reported=float(pub.loc[b, 'accuracy'])))
    chk = pd.DataFrame(chk)
    chk['agrees'] = np.isclose(chk.from_per_claim, chk.reported, atol=5e-5)
    chk.to_csv(SUP / 'per_claim_reconciliation.csv', index=False)
    log('per-claim records against the reported table:\n' + chk.round(6).to_string(index=False))

    # ---- R1.2 and R3.4: budget against budget, within each arm
    rows = []
    for name, m in (('retrieval a system can run', base), ('oracle retrieval', orac)):
        bl = list(m.columns)
        pairs = [(bl[i], bl[i + 1]) for i in range(len(bl) - 1)] + [(bl[0], bl[-1])]
        for a, b in pairs:
            r = paired_diff(m[a].to_numpy(), m[b].to_numpy(), rng)
            rows.append(dict(arm=name, budget_a=a, budget_b=b, **r))
    bud = pd.DataFrame(rows)
    bud.to_csv(SUP / 'paired_budget_comparisons.csv', index=False)
    log('\nbudget against budget, paired (difference is the larger budget minus the smaller):\n'
        + bud[['arm', 'budget_a', 'budget_b', 'difference', 'ci_low', 'ci_high',
               'mcnemar_discordant', 'mcnemar_p']].round(4).to_string(index=False))

    # ---- R1.2: oracle against the retrieval a system can run, at each budget
    rows = []
    for b in base.columns:
        r = paired_diff(base[b].to_numpy(), orac[b].to_numpy(), rng)
        rows.append(dict(budget=b, **r))
    orc = pd.DataFrame(rows)
    orc.to_csv(SUP / 'paired_oracle_vs_feasible.csv', index=False)
    log('\noracle minus feasible retrieval at each budget, paired:\n'
        + orc[['budget', 'accuracy_a', 'accuracy_b', 'difference', 'ci_low', 'ci_high',
               'mcnemar_discordant', 'mcnemar_p']].round(4).to_string(index=False))

    # ---- R3.5: what happens to the claims whose evidence arrives
    claims = load_claims(DATA / 'fever_dev.jsonl')
    store = load_page_store(OUT / 'page_store.jsonl')
    sent = V.sentence_frame(store)
    idx = np.load(OUT / 'retrieved_idx.npy')
    ev = per_claim_evidence(claims, idx, sent)
    ev.to_csv(SUP / 'evidence_per_claim.csv.gz', index=False, compression='gzip')
    held = set(base.index)
    ev_h = ev[ev.claim_id.isin(held)]
    piv = ev_h.pivot(index='claim_id', columns='budget',
                     values='complete_set_hit').reindex(base.index)
    bl = list(base.columns)
    rows = []
    for a, b in [(bl[i], bl[i + 1]) for i in range(len(bl) - 1)] + [(bl[0], bl[-1])]:
        gained = (~piv[a].to_numpy()) & piv[b].to_numpy()
        had = piv[a].to_numpy() & piv[b].to_numpy()
        never = (~piv[a].to_numpy()) & (~piv[b].to_numpy())
        for grp, mask in (('evidence arrived between the two budgets', gained),
                          ('complete evidence already retrieved', had),
                          ('complete evidence retrieved at neither budget', never)):
            if mask.sum() < 2:
                continue
            r = paired_diff(base[a].to_numpy()[mask], base[b].to_numpy()[mask], rng)
            rows.append(dict(budget_a=a, budget_b=b, group=grp, **r))
    dia = pd.DataFrame(rows)
    dia.to_csv(SUP / 'coverage_versus_accuracy.csv', index=False)
    log('\naccuracy change split by whether the complete evidence set arrived:\n'
        + dia[['budget_a', 'budget_b', 'group', 'n_claims', 'accuracy_a', 'accuracy_b',
               'difference', 'ci_low', 'ci_high', 'mcnemar_p']].round(4).to_string(index=False))

    # ---- R3.6: the feature verifier on exactly the claims the language model answered
    from evidenceloop import llm_arm as LA
    r2 = np.random.default_rng(LA.SEED)
    perm = r2.permutation(len(claims))
    test = perm[int(len(claims) * 0.5):]
    lab = claims.label.to_numpy()
    pick, per = [], LA.N_CLAIMS // len(LA.LABELS)
    for L in LA.LABELS:
        cand = test[lab[test] == L]
        pick.append(r2.choice(cand, size=min(per, len(cand)), replace=False))
    sel = np.sort(np.concatenate(pick))
    sel_ids = claims.claim_id.to_numpy()[sel]
    log(f'\nlanguage-model subsample reproduced: {len(sel)} claims, '
        + ', '.join(f'{L} {int((lab[sel] == L).sum())}' for L in LA.LABELS))
    assert set(sel_ids) <= held, 'the subsample must come from the held-out half'
    sub = base.loc[sorted(sel_ids)]
    rows = []
    for b in LA.BUDGETS:
        a_full = base[b].to_numpy()
        a_sub = sub[b].to_numpy()
        i = rng.integers(0, len(a_sub), size=(BOOT, len(a_sub)))
        lo, hi = np.percentile(a_sub[i].mean(axis=1), [2.5, 97.5])
        rows.append(dict(budget=b, n_matched=len(a_sub),
                         feature_verifier_matched=float(a_sub.mean()),
                         matched_ci_low=float(lo), matched_ci_high=float(hi),
                         feature_verifier_full_sample=float(a_full.mean()),
                         n_full_sample=len(a_full)))
    mat = pd.DataFrame(rows)
    mat.to_csv(SUP / 'matched_subsample_feature_verifier.csv', index=False)
    pd.DataFrame(dict(claim_id=sorted(sel_ids))).to_csv(
        SUP / 'language_model_subsample_claim_ids.csv', index=False)
    log('feature verifier on the matched claims against the full held-out half:\n'
        + mat.round(4).to_string(index=False))

    # ---- R3.6 continued: the two verifiers on the same claims, tested pairwise. This needs
    # the language model's per-claim records, which only the re-run retains.
    # only the re-run retained per-claim records; the canonical llm_arm_* files are the
    # run the paper reports, which did not
    lp = OUT / '_oct29_rerun' / 'llm_arm_predictions.csv.gz'
    if lp.exists():
        L = pd.read_csv(lp)
        cid = claims.claim_id.to_numpy()
        L['claim_id'] = cid[L.claim_index.to_numpy()]
        L['ok'] = (L.pred == L.true).astype(float)
        rows = []
        for b in LA.BUDGETS:
            d = L[L.budget == b].set_index('claim_id').reindex(sorted(sel_ids))
            assert d.ok.notna().all(), f'budget {b}: a subsample claim has no completion'
            a = base.loc[sorted(sel_ids), b].to_numpy()        # feature verifier
            bb = d.ok.to_numpy().astype(int)                   # language model
            rows.append(dict(budget=b, **paired_diff(a, bb, rng)))
        mc = pd.DataFrame(rows)
        mc.to_csv(SUP / 'matched_mcnemar_language_model.csv', index=False)
        log('\nlanguage model minus feature verifier on the same 198 claims, paired:\n'
            + mc[['budget', 'accuracy_a', 'accuracy_b', 'difference', 'ci_low', 'ci_high',
                  'mcnemar_b_wrong_to_right', 'mcnemar_c_right_to_wrong',
                  'mcnemar_p']].round(4).to_string(index=False))
    else:
        log('\nlanguage model per-claim records absent; matched test NOT COMPUTED')

    (SUP / 'paired_stats_manifest.json').write_text(json.dumps(dict(
        status='COMPLETED', seed=SEED, bootstrap_resamples=BOOT,
        pairing='every comparison is between two conditions scored on the same held-out '
                'claims, so the bootstrap resamples claims and McNemar is computed on the '
                'discordant pairs',
        inputs=['predictions.csv.gz', 'predictions_oracle.csv.gz',
                'verification_by_budget.csv', 'retrieved_idx.npy', 'fever_dev.jsonl'],
        note='These analyses were added for the ACLing 2026 review. They introduce no new '
             'measurement: the accuracies are re-derived from the retained per-claim records '
             'and reconciled against the reported table.'), indent=2))
    log('\nPAIRED STATS COMPLETED')


if __name__ == '__main__':
    main()
