"""Record how many sentences the retrieval pool actually holds.

The manuscript reported the pool's sentence count as a typed literal while every other
pool figure beside it came from the run manifest, which records pages but never sentences.
A typed number cannot be traced, so the review could not confirm it.

This counts the retained page store with the retrieval code's own filter, the `len(s) > 15`
test in verify.sentence_frame and writes the result where the manuscript can read it. The
count is the one the retrieval index was built over, so it is the number the paper means.
"""
from __future__ import annotations

import json
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
STORE = PROJ / 'results_real' / 'page_store.jsonl'
OUT = PROJ / 'results_real' / 'pool_sentence_count.json'
MIN_LEN = 15                      # verify.sentence_frame keeps a sentence when len(s) > 15


def main() -> None:
    pages = kept = total = 0
    with open(STORE) as f:
        for line in f:
            pages += 1
            for _, s in json.loads(line)['lines']:
                total += 1
                if len(s) > MIN_LEN:
                    kept += 1
    rec = dict(
        pages=pages, sentences_in_store=total, sentences_in_retrieval_pool=kept,
        filter=f'a stored line is a retrieval sentence when len(s) > {MIN_LEN}, '
               f'which is the test in verify.sentence_frame',
        source=str(STORE.relative_to(PROJ)),
        meaning='the sentence count the retrieval index was built over and the pool size '
                'every retrieval cost in this paper is measured against')
    OUT.write_text(json.dumps(rec, indent=1) + '\n')
    print(f'pages {pages:,}  stored lines {total:,}  retrieval sentences {kept:,}')


if __name__ == '__main__':
    main()
