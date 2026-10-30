"""Extract the retrieval pool: every page the claim set cites, plus sampled distractors.

The full encyclopaedia snapshot is 5.4 million pages. Indexing all of it is not necessary to
measure what reading costs, but a pool of only the cited pages would make retrieval trivially
easy. The pool is therefore every cited page plus a declared random sample of the rest and
the sampling is recorded: a smaller pool makes retrieval EASIER than reality, so every cost
reported from it is a lower bound.
"""
from __future__ import annotations
import json, sys, time, zipfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidenceloop.corpus import load_claims, norm_title
ROOT = Path(__file__).resolve().parents[2]
DATA, OUT = ROOT/'data_real', ROOT/'results_real'; OUT.mkdir(exist_ok=True)
N_DISTRACTORS = 120_000
SEED = 0


def log(m):
    line = f'{time.strftime("%H:%M:%S")}  {m}'
    print(line, flush=True); open(OUT/'store.log', 'a').write(line + '\n')


def main():
    claims = load_claims(DATA/'fever_dev.jsonl')
    wanted = set()
    for p in claims.gold_pages:
        wanted.update(p)
    log(f'claims {len(claims):,}; cited pages {len(wanted):,}; '
        f'sampling {N_DISTRACTORS:,} distractors')
    rng = np.random.default_rng(SEED)
    kept, seen, taken_d = {}, 0, 0
    with zipfile.ZipFile(DATA/'wiki-pages.zip') as z:
        members = sorted(n for n in z.namelist()
                         if n.startswith('wiki-pages/') and n.endswith('.jsonl'))
        # a page is kept if it is cited or with probability chosen to reach the target
        total_pages_est = 5_400_000
        p_keep = min(1.0, N_DISTRACTORS / total_pages_est)
        for i, m in enumerate(members, 1):
            with z.open(m) as fh:
                for line in fh:
                    seen += 1
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    t = norm_title(d.get('id', ''))
                    if not t:
                        continue
                    cited = t in wanted
                    if not cited and (taken_d >= N_DISTRACTORS or rng.random() > p_keep):
                        continue
                    if t in kept:
                        continue
                    lines = []
                    for ln in (d.get('lines') or '').split('\n'):
                        parts = ln.split('\t')
                        if len(parts) >= 2 and parts[0].isdigit() and parts[1].strip():
                            lines.append((int(parts[0]), parts[1]))
                    if not lines:
                        continue
                    kept[t] = lines
                    if not cited:
                        taken_d += 1
            if i % 20 == 0:
                log(f'  shard {i}/{len(members)}: {len(kept):,} pages kept '
                    f'({taken_d:,} distractors), {seen:,} scanned')
    with open(OUT/'page_store.jsonl', 'w') as fh:
        for t, lines in kept.items():
            fh.write(json.dumps(dict(title=t, lines=lines)) + '\n')
    meta = dict(pages_cited_wanted=len(wanted),
                pages_cited_found=sum(1 for t in kept if t in wanted),
                distractors=taken_d, pages_total=len(kept), lines_scanned=seen,
                n_distractors_target=N_DISTRACTORS, seed=SEED,
                note=('The retrieval pool is every cited page plus a random sample of the '
                      'rest. A smaller pool makes retrieval easier than the full snapshot, so '
                      'every cost measured on it is a lower bound.'))
    (OUT/'page_store_meta.json').write_text(json.dumps(meta, indent=2))
    log('STORE COMPLETE: ' + json.dumps(meta))


if __name__ == '__main__':
    main()
