"""Claims, the evidence that settles them and what reading that evidence costs.

The corpus is FEVER: human-written claims against an encyclopaedia snapshot, each labelled
SUPPORTS, REFUTES or NOT ENOUGH INFO, with the sentences a human annotator used recorded for
the first two. The third label is the one this study is about: a claim a responsible system
should withhold rather than assert.

Cost here is counted in sentences read. A verifier that must read the whole encyclopaedia to
decide anything is useless; one that decides without reading is unaccountable. The question is
what reading buys and whether it buys the same thing for a claim that can be settled as for
one that cannot.

No duration of this analysis is reported anywhere.
"""
from __future__ import annotations
import json, re, unicodedata, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

LABELS = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']


def norm_title(t: str) -> str:
    return unicodedata.normalize('NFC', str(t)).replace(' ', '_')


def load_claims(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            ev = d.get('evidence') or []
            pages, sents, groups = set(), set(), []
            for group in ev:
                g = set()
                for item in group:
                    if len(item) >= 4 and item[2]:
                        key = (norm_title(item[2]), item[3])
                        pages.add(key[0]); sents.add(key); g.add(key)
                if g:
                    groups.append(sorted(g))
            # A FEVER claim may be settled by any ONE complete annotator group. The flattened
            # sentence set cannot express that, so the groups are kept beside it: reviewer 3 of
            # ACLing 2026 asked for complete-set retrieval rather than at-least-one-sentence.
            rows.append(dict(claim_id=d['id'], claim=d['claim'], label=d['label'],
                             verifiable=d.get('verifiable'),
                             gold_pages=sorted(pages),
                             gold_sentences=sorted(sents),
                             gold_groups=groups,
                             n_gold_pages=len(pages), n_gold_sentences=len(sents),
                             n_gold_groups=len(groups)))
    return pd.DataFrame(rows)


def wanted_pages(claims: pd.DataFrame, extra: int = 0) -> set:
    w = set()
    for p in claims.gold_pages:
        w.update(p)
    return w


def build_page_store(zip_path: Path, wanted: set, out_path: Path,
                     log=print) -> dict:
    """Extract only the pages the claim set refers to, with their sentence splits."""
    kept, seen = {}, 0
    with zipfile.ZipFile(zip_path) as z:
        members = [n for n in z.namelist()
                   if n.startswith('wiki-pages/') and n.endswith('.jsonl')]
        for i, m in enumerate(sorted(members), 1):
            with z.open(m) as fh:
                for line in fh:
                    seen += 1
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    t = norm_title(d.get('id', ''))
                    if t in wanted and t not in kept:
                        lines = []
                        for ln in (d.get('lines') or '').split('\n'):
                            parts = ln.split('\t')
                            if len(parts) >= 2 and parts[0].isdigit():
                                lines.append((int(parts[0]), parts[1]))
                        kept[t] = lines
            if i % 40 == 0:
                log(f'  scanned {i} of {len(members)} shards, {len(kept):,} pages kept')
            if len(kept) >= len(wanted):
                break
    with open(out_path, 'w') as fh:
        for t, lines in kept.items():
            fh.write(json.dumps(dict(title=t, lines=lines)) + '\n')
    return dict(pages_wanted=len(wanted), pages_found=len(kept), lines_scanned=seen)


def load_page_store(path: Path) -> dict:
    store = {}
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            store[d['title']] = {int(i): s for i, s in d['lines']}
    return store
