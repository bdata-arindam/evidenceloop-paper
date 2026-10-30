"""Figures for EVIDENCELOOP. Cost is sentences read; no duration is reported anywhere."""
from __future__ import annotations
from pathlib import Path
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np, pandas as pd
PAL=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#4a3aa7']
SURFACE,INK,INK2,INK3,GRID='#fcfcfb','#0b0b0b','#52514e','#8a8982','#e4e3dd'
LAB=['SUPPORTS','REFUTES','NOT ENOUGH INFO']

def apply_style():
    mpl.rcParams.update({'figure.facecolor':SURFACE,'axes.facecolor':SURFACE,
      'savefig.facecolor':SURFACE,'savefig.dpi':400,'figure.dpi':140,
      'font.family':'DejaVu Sans','font.size':8.5,'axes.edgecolor':INK3,
      'axes.linewidth':0.6,'axes.labelcolor':INK,'axes.titlecolor':INK,
      'axes.titlesize':9.5,'axes.titleweight':'bold','axes.titlelocation':'left',
      'axes.titlepad':8,'axes.labelsize':8.5,'axes.spines.top':False,
      'axes.spines.right':False,'xtick.color':INK2,'ytick.color':INK2,
      'xtick.labelsize':8,'ytick.labelsize':8,'grid.color':GRID,'grid.linewidth':0.6,
      'legend.frameon':False,'legend.fontsize':8,'legend.labelcolor':INK2,
      'lines.linewidth':2.0,'figure.constrained_layout.use':True})


def fig_cost(rec: pd.DataFrame, res: pd.DataFrame, out: Path):
    """Reading buys evidence. It does not buy a decision."""
    apply_style(); fig,ax=plt.subplots(figsize=(6.8,3.2))
    ax.plot(rec.budget,rec.sentence_recall,color=PAL[0],marker='o',markersize=5,lw=2.2,
            markeredgecolor=SURFACE,markeredgewidth=0.9,zorder=4,
            label='the annotator\'s own sentence is among those read')
    ax.plot(rec.budget,rec.page_recall,color=PAL[2],marker='s',markersize=5,lw=2.0,
            ls=(0,(5,3)),markeredgecolor=SURFACE,markeredgewidth=0.9,zorder=3,
            label='the right page is among those read')
    ax.plot(res.budget,res.accuracy,color=PAL[1],marker='^',markersize=5,lw=2.2,
            markeredgecolor=SURFACE,markeredgewidth=0.9,zorder=4,
            label='the claim is labelled correctly')
    ax.axhline(1/3,color=INK2,lw=1.1,ls=(0,(3,3)),zorder=2)
    ax.text(1.1,1/3+0.02,'chance, three balanced labels',fontsize=7.2,color=INK2)
    for r in rec.itertuples():
        if r.budget in (1,100):
            ax.annotate(f'{r.sentence_recall:.3f}',(r.budget,r.sentence_recall),
                        textcoords='offset points',xytext=(0,8),fontsize=7.0,color=PAL[0],
                        ha='center')
    for r in res.itertuples():
        if r.budget in (1,100):
            ax.annotate(f'{r.accuracy:.3f}',(r.budget,r.accuracy),textcoords='offset points',
                        xytext=(0,10 if r.budget==1 else -13),fontsize=7.0,color=PAL[1],
                        ha='center')
    ax.set_xscale('log'); ax.set_xticks(rec.budget,[str(int(b)) for b in rec.budget])
    ax.set_xlabel('Sentences read (log scale)'); ax.set_ylabel('Share of claims')
    ax.set_ylim(0,1.05)
    # Every number in the title is read off the two result frames, so a rerun that changes
    # the sweep or the fitted verifier changes the title with it rather than leaving a
    # stale figure caption behind.
    _b0,_b1=int(rec.budget.min()),int(rec.budget.max())
    _r0=float(rec[rec.budget==_b0].sentence_recall.iloc[0])
    _r1=float(rec[rec.budget==_b1].sentence_recall.iloc[0])
    _a0=float(res[res.budget==_b0].accuracy.iloc[0])
    _a1=float(res[res.budget==_b1].accuracy.iloc[0])
    ax.set_title(f'A {_b1//_b0}-fold increase in reading takes evidence recall from '
                 f'{_r0:.3f} to {_r1:.3f}'+chr(10)+
                 f'and the decision from {_a0:.3f} to {_a1:.3f}',fontsize=9.5)
    ax.grid(True,zorder=0); ax.set_axisbelow(True); ax.legend(loc='center right',fontsize=7.2)
    fig.savefig(out,bbox_inches='tight'); plt.close(fig)
    return rec.merge(res[['budget','accuracy','macro_f1']],on='budget')


def fig_oracle(res: pd.DataFrame, ores: pd.DataFrame, out: Path):
    """Even handed the annotator's own sentences, the verifier does not improve."""
    apply_style(); fig,ax=plt.subplots(figsize=(6.8,3.0))
    ax.plot(res.budget,res.accuracy,color=PAL[1],marker='o',markersize=5,lw=2.2,
            markeredgecolor=SURFACE,markeredgewidth=0.9,zorder=3,label='retrieved evidence')
    ax.plot(ores.budget,ores.accuracy,color=PAL[0],marker='s',markersize=5,lw=2.2,
            markeredgecolor=SURFACE,markeredgewidth=0.9,zorder=4,
            label='the annotator\'s own sentences placed first')
    ax.axhline(1/3,color=INK2,lw=1.1,ls=(0,(3,3)),zorder=2)
    ax.text(1.1,1/3+0.006,'chance',fontsize=7.2,color=INK2)
    gap=float((ores.accuracy-res.accuracy).mean())
    d1=float(ores.accuracy.iloc[0]-res.accuracy.iloc[0])
    ax.set_xscale('log'); ax.set_xticks(res.budget,[str(int(b)) for b in res.budget])
    ax.set_xlabel('Sentences read (log scale)'); ax.set_ylabel('Claims labelled correctly')
    ax.set_ylim(0.30,0.50)
    ax.set_title('Placing the annotator\'s own sentences first is worth '+f'{d1:+.4f}'+
                 ' at one sentence'+chr(10)+
                 'and nothing measurable from ten sentences on',fontsize=9.5)
    ax.grid(True,zorder=0); ax.set_axisbelow(True); ax.legend(loc='upper right',fontsize=7.4)
    fig.savefig(out,bbox_inches='tight'); plt.close(fig)
    return res.merge(ores,on='budget',suffixes=('_retrieved','_oracle'))


def fig_by_label(res: pd.DataFrame, out: Path):
    """Which kind of claim a larger budget helps."""
    apply_style(); fig,ax=plt.subplots(figsize=(6.8,3.0))
    for k,l in enumerate(LAB):
        ax.plot(res.budget,res[f'recall_{l}'],color=PAL[k],marker='o',markersize=5,lw=2.0,
                markeredgecolor=SURFACE,markeredgewidth=0.9,zorder=3,
                label=l.title() if l!='NOT ENOUGH INFO' else 'Not settled')
    ax.axhline(1/3,color=INK2,lw=1.1,ls=(0,(3,3)),zorder=2)
    ax.set_xscale('log'); ax.set_xticks(res.budget,[str(int(b)) for b in res.budget])
    ax.set_xlabel('Sentences read (log scale)'); ax.set_ylabel('Share of that label recovered')
    ax.set_ylim(0.30,0.55)
    ax.set_title('Not settled is the label recovered best at every budget'+chr(10)+
                 'and the one a larger reading budget helps most',fontsize=9.5)
    ax.grid(True,zorder=0); ax.set_axisbelow(True); ax.legend(loc='upper left',fontsize=7.4)
    fig.savefig(out,bbox_inches='tight'); plt.close(fig)
    return res[['budget']+[f'recall_{l}' for l in LAB]+[f'precision_{l}' for l in LAB]]


def fig_pool(meta: dict, rec: pd.DataFrame, out: Path):
    """What the retrieval pool is and why the cost reported is a lower bound."""
    apply_style(); fig,(a1,a2)=plt.subplots(1,2,figsize=(6.9,2.9),width_ratios=[1,1.3])
    vals=[meta['pages_cited_found'],meta['distractors']]
    a1.bar([0,1],vals,width=0.55,color=[PAL[0],PAL[3]],edgecolor=SURFACE,linewidth=1.0,zorder=3)
    for x,v in zip([0,1],vals): a1.text(x,v*1.03,f'{v:,}',ha='center',fontsize=8,color=INK)
    a1.set_xticks([0,1],['pages the claims cite','sampled distractors'],fontsize=7.8)
    a1.set_ylabel('Pages in the retrieval pool')
    a1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,_:f'{v:,.0f}'))
    a1.set_title(f'{meta["pages_total"]:,} pages of {meta["lines_scanned"]:,} scanned',
                 fontsize=9.5)
    a1.yaxis.grid(True,zorder=0); a1.set_axisbelow(True)

    a2.plot(rec.budget,1-rec.sentence_recall,color=PAL[1],marker='o',markersize=5,lw=2.0,
            markeredgecolor=SURFACE,markeredgewidth=0.9,zorder=3)
    a2.set_xscale('log'); a2.set_yscale('log')
    a2.set_xticks(rec.budget,[str(int(b)) for b in rec.budget])
    a2.set_xlabel('Sentences read (log scale)')
    a2.set_ylabel('Share of settleable claims whose\nevidence is still unread (log scale)')
    # Matplotlib labels only the decade on this range, which left the panel with a single
    # readable y tick and no way to read a value off it. The 1-2-5 ticks spanning the
    # plotted range are derived from the data, not chosen by hand.
    _miss=1-rec.sentence_recall; _lo,_hi=float(_miss.min()),float(_miss.max())
    _ticks=[m*10.0**e for e in range(-6,2) for m in (1,2,5) if _lo/1.5<=m*10.0**e<=_hi*1.5]
    a2.set_yticks(_ticks)
    a2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,_:f'{v:g}'))
    a2.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    a2.set_title('Evidence missed, which falls steeply',fontsize=9.5)
    a2.grid(True,zorder=0); a2.set_axisbelow(True)
    fig.suptitle('The pool is every cited page plus a sample of the rest, so retrieval here is '
                 'easier'+chr(10)+'than against the whole snapshot and every cost is a lower bound',
                 fontsize=9.5,fontweight='bold',x=0.01,ha='left')
    fig.savefig(out,bbox_inches='tight'); plt.close(fig)
    return pd.DataFrame([meta])
