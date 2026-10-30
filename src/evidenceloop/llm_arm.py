"""Execute a language model on the verification task the study measures.

Finding E-005 records that the project specification asks for an executed language model and
that none was run, so nothing in the paper is a measurement of one. This module runs one.

What it is. A locally executed instruction-tuned model reads the claim together with the
sentences the study's own retriever returned at a given budget and answers with one label. It
replaces only the verifier. The corpus, the retrieval, the budgets and the labels are the
study's own and are not touched, so the arm is comparable with the feature-based verifier
already reported.

What it is not. This is one model at one quantisation on one machine with one prompt. A
different model, a different prompt or a different decoding rule would give a different
number. The arm establishes what this model does on this task; it establishes nothing about
language models in general and the paper must not read it that way.
"""
from __future__ import annotations
import json, platform, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidenceloop.corpus import load_claims, load_page_store
from evidenceloop import verify as V

ROOT = Path(__file__).resolve().parents[2]
DATA, OUT = ROOT/'data_real', ROOT/'results_real'
# OUT is where this arm READS the study's retrieval artifacts. Where it WRITES is decided at
# run time by resolve_write_dir: a second run must not land on top of the first, because the
# manuscript cites the first and a quantised model does not reproduce bit for bit.
WRITE = OUT
MODEL = 'mlx-community/SmolLM3-3B-4bit'
# Declared before the run. The full grid is 19,998 claims at seven budgets, which this machine
# cannot carry, so a stratified subsample and four budgets spanning the range are fixed here
# and never adjusted after seeing a result.
N_CLAIMS = 200
BUDGETS = (1, 5, 20, 100)
SEED = 0
LABELS = ('SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO')
MAX_TOKENS = 8
MAX_SENT_CHARS = 320


def log(m):
    line = f'{time.strftime("%H:%M:%S")}  {m}'
    print(line, flush=True)
    (WRITE/'llm_arm.log').open('a').write(line + '\n')


def user_message(claim: str, sentences: list[str]) -> str:
    """The prompt text itself, apart from the model's own chat wrapper.

    Kept separate so the supplementary material can print the executed prompt rather than a
    transcription of it. Reviewer 1 of ACLing 2026 noted the exact prompts were not retained;
    anything published from here is by construction the text the run used.
    """
    ev = '\n'.join(f'- {s[:MAX_SENT_CHARS]}' for s in sentences) or '- (no evidence retrieved)'
    return (
        'Decide whether the evidence supports the claim, refutes it or is insufficient.\n\n'
        f'Claim: {claim}\n\nEvidence:\n{ev}\n\n'
        'Answer with exactly one of these three labels and nothing else: '
        'SUPPORTS, REFUTES, NOT ENOUGH INFO.')


def build_prompt(tok, claim: str, sentences: list[str]) -> str:
    user = user_message(claim, sentences)
    return tok.apply_chat_template([{'role': 'user', 'content': user}],
                                   add_generation_prompt=True, tokenize=False,
                                   enable_thinking=False)


def parse(out: str) -> str | None:
    """Read a label out of the completion, without guessing when it says nothing."""
    t = re.sub(r'<think>.*?</think>', ' ', out, flags=re.S).upper()
    if 'NOT ENOUGH' in t or 'INSUFFICIENT' in t or 'UNKNOWN' in t:
        return 'NOT ENOUGH INFO'
    has_s, has_r = 'SUPPORT' in t, 'REFUT' in t
    if has_s and not has_r:
        return 'SUPPORTS'
    if has_r and not has_s:
        return 'REFUTES'
    return None


def resolve_write_dir() -> Path:
    """Where this run may write without destroying an earlier one.

    results_real/ holds the run the manuscript reports. If its summary is already there,
    a fresh run is sent to a dated directory beside it instead, unless --out names one or
    --force says to overwrite deliberately.
    """
    argv = sys.argv[1:]
    if '--out' in argv:
        d = ROOT / argv[argv.index('--out') + 1]
    elif '--force' in argv or not (OUT/'llm_arm_summary.csv').exists():
        d = OUT
    else:
        d = OUT / f'_rerun_{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}'
        print(f'results_real/llm_arm_summary.csv exists and is the run of record; '
              f'writing this run to {d.relative_to(ROOT)} instead. '
              f'Pass --force to overwrite it deliberately.', flush=True)
    d.mkdir(parents=True, exist_ok=True)
    return d


def main():
    from mlx_lm import load, generate
    global WRITE
    WRITE = resolve_write_dir()
    OUT.mkdir(exist_ok=True)
    run_id = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    log(f'RUN {run_id} start, model {MODEL}')

    claims = load_claims(DATA/'fever_dev.jsonl')
    store = load_page_store(OUT/'page_store.jsonl')
    sent = V.sentence_frame(store)
    idx = np.load(OUT/'retrieved_idx.npy')
    texts = sent.sentence.to_numpy()
    log(f'claims {len(claims):,}  sentences {len(sent):,}  retrieval {idx.shape}')

    # the same held-out half the feature verifier was scored on, then a stratified subsample
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(claims))
    test = perm[int(len(claims)*0.5):]
    lab = claims.label.to_numpy()
    pick = []
    per = N_CLAIMS // len(LABELS)
    for L in LABELS:
        cand = test[lab[test] == L]
        pick.append(rng.choice(cand, size=min(per, len(cand)), replace=False))
    sel = np.sort(np.concatenate(pick))
    log(f'subsample: {len(sel)} claims, '
        + ', '.join(f'{L} {int((lab[sel]==L).sum())}' for L in LABELS))

    model, tok = load(MODEL)
    rows = []
    for b in BUDGETS:
        for i in sel:
            s = [texts[j] for j in idx[i, :b] if 0 <= j < len(texts)]
            out = generate(model, tok, prompt=build_prompt(tok, claims.claim.iloc[i], s),
                           max_tokens=MAX_TOKENS, verbose=False)
            rows.append(dict(budget=b, claim_index=int(i), true=lab[i],
                             raw=out.strip()[:200], pred=parse(out)))
        d = pd.DataFrame([r for r in rows if r['budget'] == b])
        ans = d.dropna(subset=['pred'])
        acc = float((ans.pred == ans.true).mean()) if len(ans) else float('nan')
        acc_all = float((d.pred == d.true).mean())
        log(f'  budget {b:>3}: answered {len(ans)}/{len(d)}  '
            f'accuracy among answered {acc:.4f}  over all {acc_all:.4f}')
    r = pd.DataFrame(rows)
    r.to_csv(WRITE/'llm_arm_predictions.csv.gz', index=False, compression='gzip')

    summ = []
    for b in BUDGETS:
        d = r[r.budget == b]
        ans = d.dropna(subset=['pred'])
        row = dict(budget=b, claims=len(d), answered=len(ans),
                   unparsed=int(d.pred.isna().sum()),
                   accuracy_over_all=float((d.pred == d.true).mean()),
                   accuracy_among_answered=(float((ans.pred == ans.true).mean())
                                            if len(ans) else float('nan')))
        for L in LABELS:
            m = d.true == L
            row[f'recall_{L}'] = float((d.pred[m] == L).mean()) if m.any() else float('nan')
            row[f'predicted_{L}'] = float((d.pred == L).mean())
        summ.append(row)
    s = pd.DataFrame(summ)
    s.to_csv(WRITE/'llm_arm_summary.csv', index=False)
    log('summary:\n' + s.round(4).to_string(index=False))

    feat = pd.read_csv(OUT/'verification_by_budget.csv').set_index('budget')
    cmp = [dict(budget=b,
                language_model=float(s[s.budget == b].accuracy_over_all.iloc[0]),
                feature_verifier=float(feat.loc[b, 'accuracy']),
                chance=1/3) for b in BUDGETS]
    pd.DataFrame(cmp).to_csv(WRITE/'llm_arm_vs_features.csv', index=False)
    log('against the feature verifier on the full test half:\n'
        + pd.DataFrame(cmp).round(4).to_string(index=False))

    man = dict(run_id=run_id, status='COMPLETED',
               arm='language model verifier',
               model=MODEL, runtime='mlx on the local GPU', decoding='greedy',
               max_new_tokens=MAX_TOKENS, prompt='single turn, thinking disabled, '
                                                 'one label requested and nothing else',
               subsample=dict(n_claims=int(len(sel)), budgets=list(BUDGETS), seed=SEED,
                              drawn_from='the same held-out half the feature verifier was '
                                         'scored on, stratified by label'),
               summary=s.to_dict('records'), comparison=cmp,
               environment=dict(python=platform.python_version(), platform=platform.platform(),
                                numpy=np.__version__, pandas=pd.__version__),
               limitations=[
                 'One model at one quantisation on one machine with one prompt and greedy '
                 'decoding. A different model, prompt or decoding rule would give a different '
                 'number. Nothing here is a measurement of language models in general.',
                 'The grid is a stratified subsample of 200 claims at four budgets, declared '
                 'before the run because the full 19,998 by seven grid is beyond this machine. '
                 'The comparison against the feature verifier is therefore between a subsample '
                 'and a full test half and is reported as such.',
                 'A completion that names no label is counted as unanswered rather than '
                 'guessed. Both the rate over all claims and the rate among answered claims '
                 'are reported so neither flatters the model.',
                 'The retrieval, the budgets, the corpus and the labels are the study\'s own '
                 'and were not re-tuned for this arm.',
                 'The model may have seen this corpus during its own training. Nothing here '
                 'controls for that, so a high score could reflect memorisation rather than '
                 'reading the evidence supplied.'])
    (WRITE/'llm_arm_manifest.json').write_text(json.dumps(man, indent=2))
    log(f'llm_arm_manifest.json written; RUN {run_id} COMPLETED')


if __name__ == '__main__':
    main()
