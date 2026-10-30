"""Reading evidence and what reading more does to each kind of claim.

A verifier retrieves sentences and then decides: assert the claim, refute it or withhold it
because the evidence does not settle it. Cost is the number of sentences read.

The asymmetry this module measures: reading more can confirm that evidence exists, but it can
never confirm that evidence does not. A claim that should be withheld looks the same after ten
sentences and after a hundred, except that the hundred contain more near-misses.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd

BUDGETS = [1, 2, 5, 10, 20, 50, 100]
LABELS = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']


def sentence_frame(store: dict) -> pd.DataFrame:
    rows = []
    for title, lines in store.items():
        for i, s in lines.items():
            if len(s) > 15:
                rows.append((title, int(i), s))
    return pd.DataFrame(rows, columns=['title', 'line', 'sentence'])


def build_index(sent: pd.DataFrame, max_features: int = 400_000):
    from sklearn.feature_extraction.text import TfidfVectorizer
    text = (sent.title.str.replace('_', ' ', regex=False) + ' . ' + sent.sentence)
    v = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=max_features,
                        strip_accents='unicode', ngram_range=(1, 1), dtype=np.float32)
    X = v.fit_transform(text)
    return v, X


def retrieve(v, X, claims: list, top: int):
    from sklearn.preprocessing import normalize
    Q = v.transform(claims)
    out_idx, out_score = [], []
    step = 256
    for s in range(0, Q.shape[0], step):
        sims = (Q[s:s+step] @ X.T).toarray()
        idx = np.argpartition(-sims, top-1, axis=1)[:, :top]
        srt = np.take_along_axis(sims, idx, 1)
        order = np.argsort(-srt, axis=1)
        out_idx.append(np.take_along_axis(idx, order, 1))
        out_score.append(np.take_along_axis(srt, order, 1))
    return np.vstack(out_idx), np.vstack(out_score)


def evidence_recall(claims: pd.DataFrame, idx: np.ndarray, sent: pd.DataFrame,
                    budgets=BUDGETS) -> pd.DataFrame:
    key = {(t, l): i for i, (t, l) in enumerate(zip(sent.title, sent.line))}
    gold = [set(key[(t, l)] for t, l in g if (t, l) in key) for g in claims.gold_sentences]
    groups = claims.gold_groups.tolist() if 'gold_groups' in claims else [[] for _ in range(len(claims))]
    rows = []
    for b in budgets:
        hit, page_hit, full, n = 0, 0, 0, 0
        gold_pages = claims.gold_pages.tolist()
        for r in range(len(claims)):
            if not gold[r]:
                continue
            n += 1
            top = set(idx[r, :b].tolist())
            if top & gold[r]:
                hit += 1
            # at least one COMPLETE annotator group inside the retrieved set, which is what the
            # FEVER score requires and what the at-least-one measure above does not test
            if any(set(key[(ti, li)] for ti, li in g if (ti, li) in key) <= top
                   and all((ti, li) in key for ti, li in g)
                   for g in groups[r]):
                full += 1
            titles = set(sent.title.to_numpy()[idx[r, :b]])
            if titles & set(gold_pages[r]):
                page_hit += 1
        rows.append(dict(budget=b, verifiable_claims=n,
                         sentence_recall=hit/n if n else np.nan,
                         complete_set_recall=full/n if n else np.nan,
                         page_recall=page_hit/n if n else np.nan))
    return pd.DataFrame(rows)


TOKEN = re.compile(r'[a-z0-9]+')


def features(claims: pd.DataFrame, idx: np.ndarray, score: np.ndarray,
             sent: pd.DataFrame, budget: int) -> pd.DataFrame:
    s_text = sent.sentence.to_numpy(); s_title = sent.title.to_numpy()
    rows = []
    for r, claim in enumerate(claims.claim.to_numpy()):
        ct = set(TOKEN.findall(claim.lower()))
        sc = score[r, :budget]
        cover, titles = [], set()
        for j in range(budget):
            st = set(TOKEN.findall(s_text[idx[r, j]].lower()))
            cover.append(len(ct & st)/max(len(ct), 1))
            titles.add(s_title[idx[r, j]])
        cover = np.array(cover)
        rows.append(dict(
            top_score=float(sc[0]), mean_score=float(sc.mean()),
            score_gap=float(sc[0]-sc[min(1, budget-1)]),
            score_decay=float(sc[0]-sc[-1]),
            max_cover=float(cover.max()), mean_cover=float(cover.mean()),
            cover_gap=float(cover.max()-np.median(cover)),
            n_titles=len(titles), claim_tokens=len(ct),
            above_half_cover=int((cover >= 0.5).sum()),
            budget=budget))
    return pd.DataFrame(rows)


FEATURES = ['top_score', 'mean_score', 'score_gap', 'score_decay', 'max_cover',
            'mean_cover', 'cover_gap', 'n_titles', 'claim_tokens', 'above_half_cover']
