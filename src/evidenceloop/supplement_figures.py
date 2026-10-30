"""Figures for the supplementary material added in the ACLing 2026 review.

Two things the main paper states in prose and has no room to draw. First, the two
definitions of evidence recall: at least one annotator sentence retrieved, against a
complete annotator set retrieved, which is what the shared task's own score requires.
Second, where the accuracy gained by reading more actually comes from, split by whether
the complete evidence set arrived between the two budgets being compared.

Cost is sentences read; no duration is reported anywhere.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidenceloop.figures import apply_style, PAL, SURFACE, INK, INK2

ROOT = Path(__file__).resolve().parents[2]
OUT, SUP = ROOT / 'results_real', ROOT / 'results_real' / 'supplement'
FIG = ROOT / 'figures'


def fig_recall_definitions(rec: pd.DataFrame, out: Path):
    apply_style()
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    ax.plot(rec.budget, rec.sentence_recall, color=PAL[0], marker='o', markersize=5, lw=2.2,
            markeredgecolor=SURFACE, markeredgewidth=0.9, zorder=4,
            label='at least one annotator sentence retrieved')
    ax.plot(rec.budget, rec.complete_set_recall, color=PAL[5], marker='D', markersize=4.5,
            lw=2.2, ls=(0, (5, 3)), markeredgecolor=SURFACE, markeredgewidth=0.9, zorder=3,
            label='a complete annotator set retrieved')
    for r in rec.itertuples():
        if r.budget in (1, 100):
            ax.annotate(f'{r.sentence_recall:.3f}', (r.budget, r.sentence_recall),
                        textcoords='offset points', xytext=(0, 8), fontsize=7.0,
                        color=PAL[0], ha='center')
            ax.annotate(f'{r.complete_set_recall:.3f}', (r.budget, r.complete_set_recall),
                        textcoords='offset points', xytext=(0, -13), fontsize=7.0,
                        color=PAL[5], ha='center')
    ax.set_xscale('log'); ax.set_xticks(list(rec.budget))
    ax.set_xticklabels([str(b) for b in rec.budget])
    ax.set_xlabel('sentences read'); ax.set_ylabel('share of settleable claims')
    ax.set_title('Evidence recall under the two definitions')
    ax.grid(axis='y'); ax.set_ylim(0.48, 1.02); ax.legend(loc='lower right')
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)


def fig_where_accuracy_comes_from(dia: pd.DataFrame, out: Path):
    """Consecutive-budget accuracy change, split by whether the evidence arrived."""
    apply_style()
    groups = ['evidence arrived between the two budgets',
              'complete evidence already retrieved',
              'complete evidence retrieved at neither budget']
    short = ['evidence arrived', 'already retrieved', 'never retrieved']
    cols = [PAL[2], PAL[0], PAL[1]]
    d = dia[(dia.budget_a == 1) & (dia.budget_b == 100)]
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    x = np.arange(len(groups))
    vals = [float(d[d.group == g].difference.iloc[0]) for g in groups]
    lo = [float(d[d.group == g].ci_low.iloc[0]) for g in groups]
    hi = [float(d[d.group == g].ci_high.iloc[0]) for g in groups]
    n = [int(d[d.group == g].n_claims.iloc[0]) for g in groups]
    err = np.vstack([np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)])
    ax.bar(x, vals, width=0.52, color=cols, zorder=3)
    ax.errorbar(x, vals, yerr=err, fmt='none', ecolor=INK2, elinewidth=1.1,
                capsize=4, zorder=4)
    ax.axhline(0, color=INK, lw=0.9, zorder=2)
    for i, (v, c) in enumerate(zip(vals, n)):
        # anchor the label to the end of the interval, not the top of the bar, or the
        # text sits on top of the whisker
        anchor = hi[i] if v >= 0 else lo[i]
        ax.annotate(f'{v:+.4f}\n{c:,} claims', (i, anchor),
                    textcoords='offset points',
                    xytext=(0, 7 if v >= 0 else -24), fontsize=7.4,
                    color=INK2, ha='center')
    ax.set_xticks(x); ax.set_xticklabels(short)
    ax.set_ylabel('accuracy at 100 minus accuracy at 1')
    ax.set_title('Where the accuracy gained by reading more comes from')
    ax.grid(axis='y'); ax.set_ylim(min(lo) - 0.022, max(hi) + 0.030)
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)


def fig_seed_stability(seeds: pd.DataFrame, orc: pd.DataFrame, out: Path):
    """What survives resampling: the curve on the left, the oracle gap on the right."""
    apply_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    w = seeds.pivot_table(index=['seed', 'budget'], columns='arm', values='accuracy')
    af = w['annotator-first'].unstack()
    for s in af.index:
        a1.plot(af.columns, af.loc[s], color=PAL[0], lw=0.8, alpha=0.30, zorder=2)
    a1.plot(af.columns, af.mean(), color=PAL[0], lw=2.4, marker='o', markersize=4.5,
            markeredgecolor=SURFACE, markeredgewidth=0.9, zorder=4, label='mean of 20 seeds')
    a1.plot(af.columns, af.loc[0], color=PAL[1], lw=1.8, ls=(0, (4, 2)), marker='s',
            markersize=4, markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=5,
            label='the reported split')
    a1.set_xscale('log'); a1.set_xticks(list(af.columns))
    a1.set_xticklabels([str(b) for b in af.columns])
    a1.set_xlabel('sentences read'); a1.set_ylabel('claims labelled correctly')
    a1.set_title('Annotator-first arm across 20 seeds', fontsize=9)
    a1.grid(axis='y'); a1.legend(loc='lower right', fontsize=7)

    gap = (w['annotator-first'] - w['feasible']).unstack()
    x = np.arange(len(gap.columns))
    for i, b in enumerate(gap.columns):
        a2.scatter(np.full(len(gap), i) + np.linspace(-.16, .16, len(gap)), gap[b],
                   s=9, color=PAL[0], alpha=0.45, zorder=3, linewidths=0)
    a2.plot(x, gap.mean(), color=PAL[0], lw=2.2, marker='o', markersize=4.5,
            markeredgecolor=SURFACE, markeredgewidth=0.9, zorder=5, label='mean of 20 seeds')
    a2.plot(x, [orc.loc[b, 'difference'] for b in gap.columns], color=PAL[1], lw=1.8,
            ls=(0, (4, 2)), marker='s', markersize=4, markeredgecolor=SURFACE,
            markeredgewidth=0.8, zorder=6, label='the reported split')
    a2.axhline(0, color=INK, lw=0.9, zorder=2)
    a2.set_xticks(x); a2.set_xticklabels([str(b) for b in gap.columns])
    a2.set_xlabel('sentences read')
    a2.set_ylabel('annotator-first minus feasible')
    a2.set_title('The oracle gain across 20 seeds', fontsize=9)
    a2.grid(axis='y'); a2.legend(loc='upper right', fontsize=7)
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)


def main():
    rec = pd.read_csv(OUT / 'evidence_recall.csv')
    dia = pd.read_csv(SUP / 'coverage_versus_accuracy.csv')
    fig_recall_definitions(rec, FIG / 'figure_S1_recall_definitions.png')
    rec[['budget', 'verifiable_claims', 'sentence_recall', 'complete_set_recall',
         'page_recall']].to_csv(FIG / 'figure_S1_recall_definitions_table.csv', index=False)
    fig_where_accuracy_comes_from(dia, FIG / 'figure_S2_where_accuracy_comes_from.png')
    seeds = pd.read_csv(SUP / 'seed_stability.csv')
    orc = pd.read_csv(SUP / 'paired_oracle_vs_feasible.csv').set_index('budget')
    fig_seed_stability(seeds, orc, FIG / 'figure_S5_seed_stability.png')
    seeds.to_csv(FIG / 'figure_S5_seed_stability_table.csv', index=False)
    dia[(dia.budget_a == 1) & (dia.budget_b == 100)].to_csv(
        FIG / 'figure_S2_where_accuracy_comes_from_table.csv', index=False)
    print('figure_S1_recall_definitions.png')
    print('figure_S2_where_accuracy_comes_from.png')
    print('figure_S5_seed_stability.png')


if __name__ == '__main__':
    main()
