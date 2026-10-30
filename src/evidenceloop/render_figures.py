from pathlib import Path
import json, sys
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evidenceloop import figures as F
ROOT=Path(__file__).resolve().parents[2]; RES,FIG=ROOT/'results_real',ROOT/'figures'
FIG.mkdir(exist_ok=True)
rec=pd.read_csv(RES/'evidence_recall.csv'); res=pd.read_csv(RES/'verification_by_budget.csv')
ores=pd.read_csv(RES/'verification_oracle_retrieval.csv')
meta=json.loads((RES/'page_store_meta.json').read_text())
man=json.loads((RES/'run_manifest.json').read_text())
spec=[('figure_1_cost.png',lambda o:F.fig_cost(rec,res,o),'evidence_recall.csv',
       'Evidence recall and verification accuracy against sentences read.'),
      ('figure_2_oracle.png',lambda o:F.fig_oracle(res,ores,o),'verification_oracle_retrieval.csv',
       'Verification with retrieved evidence against the annotator\'s own sentences.'),
      ('figure_3_by_label.png',lambda o:F.fig_by_label(res,o),'verification_by_budget.csv',
       'Which kind of claim a larger reading budget helps.'),
      ('figure_4_pool.png',lambda o:F.fig_pool(meta,rec,o),'page_store_meta.json',
       'The retrieval pool and the share of evidence still unread at each budget.')]
reg=[]
for name,fn,src,cap in spec:
    t=fn(FIG/name); t.to_csv(FIG/name.replace('.png','_table.csv'))
    reg.append(dict(figure=name,table_view=name.replace('.png','_table.csv'),
                    generated_from=src,run_id=man['run_id'],
                    classification='EXECUTION-DERIVED RESULT',
                    generator='src/evidenceloop/figures.py',caption=cap))
    print('rendered',name)
(FIG/'FIGURE_REGISTER.json').write_text(json.dumps(reg,indent=2))
print('FIGURE_REGISTER.json written')
