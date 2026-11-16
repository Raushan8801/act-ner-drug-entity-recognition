"""Deterministic analysis of frozen DrugNER artifacts.

This module never fits a recognition model or modifies a checkpoint/prediction.
Historical development selection is exploratory; test analysis is post-hoc.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import textwrap
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from IPython.display import Image, display

from core import (ENTITY_TYPES, configure_notebook, gold_for_metrics,
                  load_development_corpus, load_processed, normalize_surface)

MODELS = {
    'O-only': 'o_only', 'Longest-match lexicon': 'longest_match_lexicon',
    'Logistic Regression': 'logistic_regression', 'Linear-chain CRF': 'linear_chain_crf',
    'BiLSTM-CRF': 'bilstm_crf', 'Biomedical Transformer': 'biomedical_transformer',
    'ACT-NER': 'act_ner',
}
FOCAL = ['Linear-chain CRF', 'Biomedical Transformer', 'ACT-NER']
BLUE, GOLD, INK = '#315C80', '#B47A19', '#252525'


def configure() -> None:
    configure_notebook()
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('styler.render.max_elements', 100000)


def check_frozen(root: Path, include_posthoc: bool = False) -> None:
    contract = json.loads((root / 'src/frozen_contract.json').read_text())
    for relative, expected in contract.items():
        if not include_posthoc and ('locked_test' in relative.lower()):
            continue
        path = root / relative
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(2**20), b''):
                h.update(chunk)
        assert h.hexdigest() == expected, relative


def table(root: Path, frame: pd.DataFrame, name: str, precision: int = 4,
          missing_note: str | None = None):
    """Save the exact analysis and return a full, non-truncated display object."""
    assert not frame.empty, 'An analytical table must contain observations.'
    assert not frame.select_dtypes(include='object').map(
        lambda v: isinstance(v, str) and not v.strip()).any().any()
    if frame.isna().any().any():
        if 'Accepted N' in frame and frame.loc[frame.isna().any(axis=1), 'Accepted N'].eq(0).all():
            missing_note = 'N/A: confidence metrics are not estimable when Accepted N = 0 (no emitted entities).'
        assert missing_note, 'Every undefined table value requires a scientific explanation.'
    frame.to_csv(root / 'artifacts' / f'{name}.csv', index=False, na_rep='N/A')
    def number(value):
        if pd.isna(value):
            return 'N/A'
        if abs(value) < .5 * 10 ** (-precision):
            return '0' if precision==0 else '0.' + '0' * precision
        return f'{value:.{precision}f}'.replace('-', '−')
    formatting = {c: number for c in frame.select_dtypes(include='floating').columns}
    styled=frame.style.hide(axis='index').format(formatting, na_rep='N/A')
    if missing_note:
        styled=styled.set_caption(missing_note).set_table_styles([
            {'selector':'caption','props':[('caption-side','bottom'),('text-align','left'),
                                           ('padding-top','0.4em')]}])
    if 'Model' in frame.columns and frame.Model.nunique()>1 and 'Subgroup' not in frame.columns:
        higher=[c for c in frame.columns if c in ('Boundary P','Boundary R','Boundary F1','Typed P','Typed R','Typed F1','Macro-F1','drug F1','brand F1','group F1','drug_n F1')]
        lower=[c for c in frame.columns if c in ('ECE','Brier','Discrete AURC')]
        if higher:styled=styled.highlight_max(subset=higher,props='font-weight: bold;')
        if lower:styled=styled.highlight_min(subset=lower,props='font-weight: bold;')
    return styled


def read_table(root: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(root / 'artifacts' / f'{name}.csv')


def span_set(frame: pd.DataFrame, typed: bool = True, gold: bool = False) -> set:
    cols = ['sentence_id', 'char_start', 'char_end']
    if typed:
        cols += ['entity_type' if gold else 'predicted_type']
    return set(frame[cols].itertuples(index=False, name=None))


def score(gold: pd.DataFrame, prediction: pd.DataFrame) -> dict:
    g, p = span_set(gold, gold=True), span_set(prediction)
    gb, pb = span_set(gold, False), span_set(prediction, False)
    tp, btp = len(g & p), len(gb & pb)
    div = lambda a, b: a / b if b else 0.0
    subtype = {}
    for kind in ENTITY_TYPES:
        gs, ps = {s for s in g if s[-1] == kind}, {s for s in p if s[-1] == kind}
        subtype[f'{kind} F1'] = div(2 * len(gs & ps), len(gs) + len(ps))
    return {'N': len(g), 'Predictions': len(p), 'TP': tp, 'FP': len(p)-tp, 'FN': len(g)-tp,
            'Boundary P': div(btp,len(pb)), 'Boundary R': div(btp,len(gb)),
            'Boundary F1': div(2*btp,len(gb)+len(pb)), 'Typed P': div(tp,len(p)),
            'Typed R': div(tp,len(g)), 'Typed F1': div(2*tp,len(g)+len(p)),
            'Macro-F1': float(np.mean(list(subtype.values()))), **subtype}


def evaluate(root: Path, split: str, model_names=None) -> tuple[pd.DataFrame, dict, tuple]:
    assert split in ('validation','locked_test')
    sentences, entities, tokens = load_processed(root, split)
    gold = gold_for_metrics(entities)
    directory = 'Validation' if split == 'validation' else 'Locked_Test'
    predictions, rows = {}, []
    for model in (model_names or list(MODELS)):
        prediction = pd.read_parquet(root / 'Predictions' / directory / (MODELS[model]+'.parquet'))
        assert prediction.entity_id.is_unique
        assert not prediction.duplicated(['sentence_id','char_start','char_end','predicted_type']).any()
        assert set(prediction.sentence_id) <= set(sentences.sentence_id)
        predictions[model] = prediction
        rows.append({'Model': model, 'Split': 'Development' if split=='validation' else 'Post-hoc test',
                     **score(gold,prediction)})
    return pd.DataFrame(rows), predictions, (sentences,gold,tokens)


def document_counts(sentences, gold, prediction):
    documents = sorted(sentences.document_id.unique())
    gg = {d:span_set(f,gold=True) for d,f in gold.groupby('document_id')}
    pp = {d:span_set(f) for d,f in prediction.groupby('document_id')}
    values = np.array([(len(gg.get(d,set()) & pp.get(d,set())),len(pp.get(d,set())),len(gg.get(d,set())))
                       for d in documents], dtype=int)
    assert values[:,1].sum()==len(span_set(prediction))
    return documents, values


def clustered_statistics(sentences, gold, predictions, resamples=2000):
    names=list(predictions)
    counts={m:document_counts(sentences,gold,p)[1] for m,p in predictions.items()}
    n=len(sentences.document_id.unique())
    rng=np.random.default_rng(20260905)
    draws=rng.integers(0,n,size=(resamples,n))
    f1=lambda c: np.divide(2*c[...,0],c[...,1]+c[...,2],out=np.zeros(c.shape[:-1]),where=(c[...,1]+c[...,2])>0)
    boot={m:f1(c[draws].sum(axis=1)) for m,c in counts.items()}
    rows=[{'Model':m,'N Documents':n,'N Gold':len(gold),'Typed F1':float(f1(c.sum(0))),
           'CI Low':np.quantile(boot[m],.025),'CI High':np.quantile(boot[m],.975),'Resamples':resamples}
          for m,c in counts.items()]
    comparisons=[]
    if 'ACT-NER' in names:
        for m in [n for n in FOCAL if n!='ACT-NER' and n in names]:
            delta=boot['ACT-NER']-boot[m]
            comparisons.append({'Comparison':f'ACT-NER vs {m}','N Documents':n,
                                'Micro-F1 difference':float(f1(counts['ACT-NER'].sum(0))-f1(counts[m].sum(0))),
                                'CI Low':np.quantile(delta,.025),'CI High':np.quantile(delta,.975),
                                'Estimand':'Corpus micro-F1; documents resampled'})
    secondary=[]
    for m in [n for n in FOCAL if n!='ACT-NER' and n in names and 'ACT-NER' in names]:
        differences=f1(counts['ACT-NER'])-f1(counts[m])
        nonzero=differences[np.abs(differences)>1e-12]
        ranks=stats.rankdata(np.abs(nonzero))
        p=float(stats.wilcoxon(nonzero,zero_method='wilcox',method='auto').pvalue) if len(nonzero) else 1.
        secondary.append({'Comparison':f'ACT-NER vs {m}','N Documents':n,'Nonzero pairs':len(nonzero),
                          'Mean document F1 difference':differences.mean(),'Median difference':np.median(differences),
                          'Rank-biserial effect':float(np.sum(ranks*np.sign(nonzero))/ranks.sum()) if len(nonzero) else 0.,
                          'p-value':p})
    secondary=pd.DataFrame(secondary)
    if len(secondary): secondary['Holm p-value']=holm(secondary['p-value'])
    return pd.DataFrame(rows),pd.DataFrame(comparisons),secondary


def holm(values):
    p=np.asarray(values);order=np.argsort(p);result=np.zeros(len(p))
    result[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return result


def calibration(gold, predictions, bins=(5,10,15,20)):
    g=span_set(gold,gold=True); rows=[];reliability=[];risk=[]
    for model,p in predictions.items():
        y=np.array([int(k in g) for k in p[['sentence_id','char_start','char_end','predicted_type']].itertuples(index=False,name=None)])
        confidence=p.calibrated_confidence.to_numpy(float)
        assert np.isfinite(confidence).all() and ((confidence>=0)&(confidence<=1)).all()
        if len(y):
            order=np.argsort(-confidence,kind='stable')
            prefix_risk=1-np.cumsum(y[order])/np.arange(1,len(y)+1)
            aurc=float(prefix_risk.mean())
        else:
            aurc=np.nan
        for nbins in bins:
            bid=np.minimum((confidence*nbins).astype(int),nbins-1)
            ece=0.
            for b in range(nbins):
                selected=bid==b
                if selected.any():
                    acc=y[selected].mean();conf=confidence[selected].mean();n=int(selected.sum())
                    ece+=n/max(len(y),1)*abs(acc-conf)
                    if nbins==10:
                        low,high=wilson(int(y[selected].sum()),n)
                        reliability.append({'Model':model,'Bin':b,'N':n,'Confidence':conf,'Accuracy':acc,'CI Low':low,'CI High':high})
            rows.append({'Model':model,'Accepted N':len(y),'Bins':nbins,'ECE':ece if len(y) else np.nan,
                         'Brier':np.mean((confidence-y)**2) if len(y) else np.nan,'Discrete AURC':aurc})
        if len(y):
            order=np.argsort(-confidence,kind='stable');correct=np.cumsum(y[order])
            for coverage in np.arange(.1,1.01,.1):
                k=max(1,math.ceil(coverage*len(y))); tp=int(correct[k-1])
                risk.append({'Model':model,'Coverage':k/len(y),'Accepted N':k,'Risk':1-tp/k,
                             'Selective Typed F1':2*tp/(k+len(g))})
    return pd.DataFrame(rows),pd.DataFrame(reliability),pd.DataFrame(risk)


def wilson(k,n):
    if n==0:return np.nan,np.nan
    z=stats.norm.ppf(.975);p=k/n;d=1+z*z/n
    return (p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/d,(p+z*z/(2*n)+z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/d


def subgroup_scores(root,sentences,gold,tokens,predictions):
    train=pd.read_parquet(root/'dataset/Processed/train_entities.parquet')
    seen=set(train.query('not discontinuous').entity_text.map(normalize_surface))
    def enrich(frame):
        f=frame.copy();f['source']=f.sentence_id.map(sentences.set_index('sentence_id').source)
        f['seen']=f.entity_text.map(normalize_surface).isin(seen)
        groups={s:t for s,t in tokens.groupby('sentence_id')}
        f['token_count']=[int(((groups[r.sentence_id].char_start<r.char_end)&(groups[r.sentence_id].char_end>r.char_start)).sum()) for r in f.itertuples()]
        return f
    gold=enrich(gold);rows=[]
    for model,p in predictions.items():
        p=enrich(p)
        selectors={'DrugBank':lambda f:f.source.eq('DrugBank'),'MedLine':lambda f:f.source.eq('MedLine'),
                   'Seen':lambda f:f.seen,'Unseen':lambda f:~f.seen,'Single token':lambda f:f.token_count.eq(1),
                   'Multi-token':lambda f:f.token_count.gt(1),'Long (4+ tokens)':lambda f:f.token_count.ge(4)}
        for name,fn in selectors.items():
            result=score(gold.loc[fn(gold)],p.loc[fn(p)])
            rows.append({'Model':model,'Subgroup':name,**result})
    return pd.DataFrame(rows)


def error_partition(gold,predictions):
    g=span_set(gold,gold=True); gb={s[:3] for s in g};type_map={s[:3]:s[-1] for s in g}
    rows=[]
    for m,p in predictions.items():
        counts={'Exact typed':0,'Wrong type':0,'Overlong':0,'Underlong':0,'Shifted overlap':0,'Spurious':0}
        by_sentence={s:list(f[['char_start','char_end']].itertuples(index=False,name=None)) for s,f in gold.groupby('sentence_id')}
        for key in span_set(p):
            s,a,b,t=key
            if key in g:category='Exact typed'
            elif key[:3] in gb:category='Wrong type'
            else:
                overlaps=[(c,d) for c,d in by_sentence.get(s,[]) if max(a,c)<min(b,d)]
                if not overlaps:category='Spurious'
                else:
                    c,d=max(overlaps,key=lambda z:min(b,z[1])-max(a,z[0]))
                    category='Overlong' if a<=c and b>=d else 'Underlong' if a>=c and b<=d else 'Shifted overlap'
            counts[category]+=1
        assert sum(counts.values())==len(span_set(p))
        rows += [{'Model':m,'Error category':k,'Count':v,'Prediction share':v/len(p) if len(p) else np.nan} for k,v in counts.items()]
    return pd.DataFrame(rows)


def corpus_analysis(root):
    documents,sentences,entities=load_development_corpus(root)
    gold=gold_for_metrics(entities).copy()
    gold['Characters']=gold.char_end-gold.char_start
    gold['Regex tokens']=gold.entity_text.map(lambda x:len(re.findall(r'\w+|[^\w\s]',x)))
    gold['Normalized']=gold.entity_text.map(normalize_surface)
    s=sentences.copy();s['Tokens']=s.text.map(lambda x:len(re.findall(r'\w+|[^\w\s]',x)))
    s['Entities']=s.sentence_id.map(gold.groupby('sentence_id').size()).fillna(0).astype(int)
    s['Characters']=s.text.str.len();s['Density']=s.Entities/s.Tokens.clip(lower=1)
    profile=documents[['document_id','source','split']].copy()
    profile=profile.merge(s.groupby('document_id').agg(Sentences=('sentence_id','size'),Tokens=('Tokens','sum'),
                          Entities=('Entities','sum'),Mean_sentence_tokens=('Tokens','mean')),on='document_id')
    subtype=pd.crosstab(gold.document_id,gold.entity_type).reindex(columns=ENTITY_TYPES,fill_value=0)
    profile=profile.merge(subtype,on='document_id',how='left').fillna(0)
    return documents,s,gold,profile


def document_source_tests(profile,permutations=1999):
    rng=np.random.default_rng(1701);source=profile.source.eq('DrugBank').to_numpy();rows=[]
    for feature in ['Mean_sentence_tokens','Entities','brand','drug_n']:
        values=profile[feature].to_numpy(float)
        if feature in ENTITY_TYPES:values=values/np.maximum(profile.Entities,1)
        observed=values[source].mean()-values[~source].mean()
        null=[]
        for _ in range(permutations):
            label=rng.permutation(source);null.append(values[label].mean()-values[~label].mean())
        pooled=np.sqrt((values[source].var(ddof=1)+values[~source].var(ddof=1))/2)
        rows.append({'Feature':feature.replace('_',' '),'N Documents':len(profile),'DrugBank mean':values[source].mean(),
                     'MedLine mean':values[~source].mean(),'SMD':observed/pooled if pooled else 0.,
                     'Permutation p':(1+np.sum(np.abs(null)>=abs(observed)))/(permutations+1)})
    frame=pd.DataFrame(rows);frame['Holm p']=holm(frame['Permutation p']);return frame


def cooccurrence_graph(sentences,gold):
    presence=pd.crosstab(gold.sentence_id,gold.entity_type).reindex(index=sentences.sentence_id,columns=ENTITY_TYPES,fill_value=0).gt(0)
    rows=[]
    for a,b in combinations(ENTITY_TYPES,2):
        both=int((presence[a]&presence[b]).sum());union=int((presence[a]|presence[b]).sum())
        rows.append({'Type A':a,'Type B':b,'Co-mention sentences':both,'Union sentences':union,
                     'Jaccard':both/union if union else 0.,'Scope':'Train only; undirected co-mention'})
    return pd.DataFrame(rows)


def duplicate_graph(sentences):
    normalized=sentences.text.map(lambda x:re.sub(r'\s+',' ',x).strip().casefold())
    f=sentences.assign(normalized=normalized)
    documents=sorted(sentences.document_id.unique()); ids={d:i for i,d in enumerate(documents)};edges=set();cross=0
    for _,group in f[f.normalized.str.len().ge(40)].groupby('normalized'):
        ds=sorted(group.document_id.unique())
        if len(ds)<2:continue
        if group.split.nunique()>1:cross+=1
        edges.update((ids[ds[0]],ids[d]) for d in ds[1:])
    rr=[i for edge in edges for i in edge];cc=[i for edge in edges for i in edge[::-1]]
    adjacency=coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(documents),len(documents)))
    ncomp,labels=connected_components(adjacency,directed=False)
    sizes=pd.Series(labels).value_counts()
    return pd.DataFrame([{'Documents':len(documents),'Components':ncomp,'Nontrivial components':int((sizes>1).sum()),
                          'Largest component':int(sizes.max()),'Shared sentence forms across splits':cross,
                          'Minimum normalized characters':40}])


def error_complementarity(gold,predictions):
    g=span_set(gold,gold=True);rows=[]
    for a,b in combinations(predictions,2):
        pa,pb=span_set(predictions[a]),span_set(predictions[b]);fa,fb=g-pa,g-pb
        rows.append({'Model A':a,'Model B':b,'Gold N':len(g),'Both missed':len(fa&fb),
                     'A only recovered':len((pa&g)-pb),'B only recovered':len((pb&g)-pa),
                     'Failure Jaccard':len(fa&fb)/max(len(fa|fb),1),
                     'Oracle union recall':len((pa|pb)&g)/len(g)})
    return pd.DataFrame(rows)


def architecture_audit(root):
    router=read_table(root,'05_router_diagnostics');overall=router.query("Grouping == 'Overall'")
    weights=overall.groupby('Expert')['Mean Weight'].mean()
    seeds=read_table(root,'05_act_seed_results');sd=seeds['Strict Typed Entity F1'].std()
    return pd.DataFrame([
        {'Claim':'All three experts participate','Observed evidence':f"Prototype mean routing weight = {weights.get('Rare-Signal Prototype',0):.6f}",'Assessment':'Not established'},
        {'Claim':'Independent repeated GroupCV','Observed evidence':'One five-fold document partition; three optimization seeds','Assessment':'Single-partition multi-seed CV'},
        {'Claim':'Out-of-selection generalization','Observed evidence':'Configurations selected and reported on the same development folds','Assessment':'Nested selection absent'},
        {'Claim':'Seed stability proves learned contribution','Observed evidence':f'Typed F1 SD = {sd:.8f}; expert anchors override learned heads','Assessment':'Decoder may mask learning'},
        {'Claim':'Isolated component effects','Observed evidence':'Prototype/router/weak-recovery switches also alter rare-signal anchor thresholds','Assessment':'Ablation confounding'},
        {'Claim':'Continual adaptation validated','Observed evidence':'Primary checkpoint has shift_triggered=False; safety protocol only','Assessment':'Not experimentally demonstrated'},
        {'Claim':'Train-only prototype memory','Observed evidence':'Prototypes initialized inside held-in development folds; not original Train corpus','Assessment':'Training scope needs qualification'},
        {'Claim':'Comparable inference efficiency','Observed evidence':'Head-only CV timing and frozen Validation comparator measurements differ in scope','Assessment':'End-to-end benchmark absent'},
    ])


def router_summary(root):
    f=read_table(root,'05_router_diagnostics').query("Grouping == 'Overall'")
    return f.groupby('Expert',as_index=False).agg(Seeds=('Seed','nunique'),Mean_weight=('Mean Weight','mean'),
                   SD_between_seed_means=('Mean Weight','std'),Mean_within_seed_SD=('SD Weight','mean'),
                   Prototype_computation_rate=('Prototype Activation Rate','mean'))


def decoder_provenance(predictions):
    act=predictions['ACT-NER'];crf=span_set(predictions['Linear-chain CRF']);transformer=span_set(predictions['Biomedical Transformer'])
    rows=[]
    for row in act.itertuples():
        key=(row.sentence_id,row.char_start,row.char_end,row.predicted_type)
        origin='Matches both experts' if key in crf and key in transformer else 'Matches CRF only' if key in crf else 'Matches Transformer only' if key in transformer else 'No exact expert match'
        rows.append(origin)
    count=pd.Series(rows).value_counts()
    return pd.DataFrame({'Output agreement':count.index,'Entities':count.values,'Share':count.values/len(act)})


def save_canvas(root,fig,name,data):
    fig.set_dpi(350);fig.canvas.draw()
    records=[]
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not t.get_text().strip():continue
        assert 18-1e-6<=t.get_fontsize()<=20+1e-6
        t.set_fontfamily('Times New Roman')
    fig.savefig(root/'figures/PNG'/f'{name}.png',dpi=350,facecolor='white')
    fig.savefig(root/'figures/PDF'/f'{name}.pdf',dpi=350,facecolor='white')
    data.to_csv(root/'artifacts'/f'figure_{name}_data.csv',index=False,na_rep='N/A')
    manifest_path=root/'Reports/figure_manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[name]={'width_inches':float(fig.get_figwidth()),'height_inches':float(fig.get_figheight()),
                    'dpi':350,'font_family':'Times New Roman','font_min':18,'font_max':20,
                    'source_csv':f'artifacts/figure_{name}_data.csv','axes':len(fig.axes)}
    manifest_path.write_text(json.dumps(manifest,indent=2))
    plt.close(fig)
    return Image(filename=str(root/'figures/PNG'/f'{name}.png'),width=round(fig.get_figwidth()*100))


def chart(root,data,name,kind,x,y,title,xlabel=None,ylabel=None,series=None,height=4.6,width=6.8,
          missing_note=None):
    """One compact scientific canvas with fixed typography and vertical legend."""
    assert not data.empty, 'An empty analytical canvas must not be published.'
    width=min(width,7.2)
    height=min(height,5.0 if series else 4.5)
    if kind=='heatmap':
        nrows,ncols=data[y].nunique(),data[x].nunique()
        width=min(7.2,max(5.8,.95*ncols+1.8))
        height=1.6+.63*nrows+.3*max(len(textwrap.wrap(str(v),11)) for v in data[x])
    if kind=='barh':height=1.6+.8*len(data)
    fig,ax=plt.subplots(figsize=(width,height),layout='constrained')
    fig.get_layout_engine().set(w_pad=.04,h_pad=.04,wspace=.02,hspace=.02)
    palette=[BLUE,GOLD,'#67713B','#965F7E','#777777','#3B757A','#A75936']
    if kind=='hist':
        ax.hist(data[x],bins='fd',color=BLUE,edgecolor='white',linewidth=.7)
        if data[x].max()>3*data[x].quantile(.75):
            ax.set_yscale('log');ax.set_ylabel('Entities (log scale)')
    elif kind=='bar':
        labels=[textwrap.fill(str(v),21) for v in data[x]];pos=np.arange(len(data))
        bars=ax.bar(pos,data[y],color=BLUE,edgecolor=INK,linewidth=.7)
        ax.set_xticks(pos,labels);ax.margins(y=.13,x=.015)
        ax.bar_label(bars,labels=[f'{v:,.0f}' for v in data[y]],padding=5,fontsize=18)
    elif kind=='barh':
        labels=[textwrap.fill(str(v),26) for v in data[x]];pos=np.arange(len(data))
        bars=ax.barh(pos,data[y],color=BLUE,edgecolor=INK,linewidth=.7)
        ax.set_yticks(pos,labels);ax.invert_yaxis();ax.margins(x=.62)
        annotations=[f'{v:.1%}\nN = {int(n)}' for v,n in zip(data[y],data.Entities)] if 'Entities' in data else [f'{v:.3g}' for v in data[y]]
        ax.bar_label(bars,labels=annotations,padding=5,fontsize=18)
    elif kind=='heatmap':
        matrix=data.pivot(index=y,columns=x,values=series)
        for axis in ('index','columns'):
            labels=list(getattr(matrix,axis))
            preferred=list(ENTITY_TYPES)+['No exact span'] if set(ENTITY_TYPES)<=set(labels) else ['O','B','I','E','S']
            if set(labels)<=set(preferred):
                matrix=matrix.reindex(**{axis:[label for label in preferred if label in labels]})
        values=matrix.to_numpy(dtype=float)
        finite=values[np.isfinite(values)]
        assert not np.isinf(values).any(), 'Infinite values require a corrected calculation.'
        if np.isnan(values).any():
            assert missing_note, 'Undefined heatmap cells require a visible scientific explanation.'
        signed=bool(len(finite) and finite.min()<0)
        cmap=mpl.colormaps['RdBu_r' if signed else 'Blues'].copy()
        cmap.set_bad('#E4E4E4')
        im=ax.imshow(np.ma.masked_invalid(values),cmap=cmap,aspect='auto',
                     **({'vmin':-1,'vmax':1} if signed else {}))
        short_labels={'Overlong':'Too\nlong','Underlong':'Too\nshort','Shifted overlap':'Shifted','Wrong type':'Wrong\ntype'}
        ax.set_xticks(range(len(matrix.columns)),[short_labels.get(str(v),textwrap.fill(str(v),11)) for v in matrix.columns])
        ax.set_yticks(range(len(matrix)),[textwrap.fill(str(v),16) for v in matrix.index])
        for i in range(len(matrix)):
            for j in range(len(matrix.columns)):
                v=matrix.iloc[i,j];rgba=im.cmap(im.norm(v));lum=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                if not np.isfinite(v):
                    label='N/A'
                elif series=='Count':
                    label=f'{v:,.0f}'
                elif 0 < abs(v) < .005:
                    label='>−0.01' if v<0 else '<0.01'
                else:
                    label=f'{v:.2f}'.replace('-','−')
                ax.text(j,i,label,ha='center',va='center',color='white' if np.isfinite(v) and lum<.5 else INK,fontsize=18)
        if np.isnan(values).any():
            note=textwrap.fill(missing_note,52)
            fig.set_figheight(fig.get_figheight()+.28*len(note.splitlines()))
            fig.supxlabel(note,fontsize=18,color=INK)
    elif kind in ('line','scatter','interval'):
        groups=list(data.groupby(series,sort=False)) if series else [('',data)]
        for i,(label,part) in enumerate(groups):
            label=textwrap.fill(str(label),25) if label else None
            if kind=='scatter':ax.scatter(part[x],part[y],s=28,alpha=.45,marker=['o','s','^','D'][i%4],color=palette[i%len(palette)],label=label)
            elif kind=='interval':
                ax.errorbar(part[x],part[y],yerr=[part[y]-part['CI Low'],part['CI High']-part[y]],fmt='o-',capsize=3,label=label,color=palette[i%len(palette)])
            else:ax.plot(part[x],part[y],marker=['o','s','^'][i%3],markersize=5,linewidth=1.8,label=label,color=palette[i%len(palette)])
        if series:
            handles,labels=ax.get_legend_handles_labels()
            fig.legend(handles,labels,loc='outside lower center',ncol=1,frameon=False,
                       borderaxespad=.45,labelspacing=.25)
        if kind=='interval' and y=='Accuracy':
            ax.plot([0,1],[0,1],linestyle='--',color=INK,alpha=.6)
            ax.set(xlim=(-.03,1.03),ylim=(-.03,1.03))
    wrapped_title='\n'.join(textwrap.fill(line,45) for line in title.splitlines())
    if kind=='barh':
        fig.suptitle(wrapped_title,fontsize=20)
    else:
        ax.set_title(wrapped_title,fontsize=20,pad=12)
    ax.set_xlabel(xlabel if xlabel is not None else x)
    ax.set_ylabel(ylabel if ylabel is not None else y)
    if kind!='heatmap':
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
        if kind=='hist' and ax.get_yscale()!='log':ax.set_ylabel(ylabel if ylabel is not None else y)
        if kind=='hist' and ax.get_yscale()=='log':
            ax.set_ylabel(f'{ylabel if ylabel is not None else y} (log scale)')
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
                lambda value,pos:f'{value:,.0f}' if value>=1 else f'{value:g}'))
        if x=='Bins':ax.set_xticks(sorted(data[x].unique()))
    return save_canvas(root,fig,name,data)


def draw_graph(root,edges,name):
    fig,ax=plt.subplots(figsize=(6.0,4.8),layout='constrained')
    positions={'drug':(-.8,.7),'brand':(.8,.7),'group':(-.8,-.7),'drug_n':(.8,-.7)}
    for i,row in edges.iterrows():
        a,b=positions[row['Type A']],positions[row['Type B']]
        ax.plot([a[0],b[0]],[a[1],b[1]],color=BLUE,alpha=.55,linewidth=1+10*row.Jaccard,zorder=1)
        fraction=.33 if i==2 else .67 if i==3 else .5
        x=a[0]+fraction*(b[0]-a[0]);y=a[1]+fraction*(b[1]-a[1])
        ax.text(x,y,f'{row.Jaccard:.3f}',ha='center',va='center',fontsize=18,
                bbox={'facecolor':'white','edgecolor':'none','pad':1.5},zorder=3)
    for node,(x,y) in positions.items():
        ax.scatter([x],[y],s=max(2600,(10*len(node)+6)**2),c='white',edgecolors=BLUE,linewidths=2,zorder=2)
        ax.text(x,y,node,ha='center',va='center',fontsize=18,zorder=3)
    ax.set(xlim=(-1.16,1.16),ylim=(-1.02,1.02));ax.axis('off')
    ax.set_title('Training entity-type co-mentions\nEdge width: sentence Jaccard',fontsize=20,pad=12)
    return save_canvas(root,fig,name,edges)


def effect_chart(root,frame,name):
    data=frame.copy();data['Comparison']=data.Comparison.str.replace('ACT-NER vs ','',regex=False)
    fig,ax=plt.subplots(figsize=(6.8,3.6),layout='constrained');pos=np.arange(len(data))
    x=data['Micro-F1 difference'];ax.errorbar(x,pos,xerr=[x-data['CI Low'],data['CI High']-x],fmt='o',capsize=6,color=BLUE)
    ax.axvline(0,color=INK,linestyle='--');ax.set_yticks(pos,[textwrap.fill(v,17) for v in data.Comparison]);ax.set_ylim(-.7,len(data)-.3)
    ax.set_title('ACT-NER paired corpus-F1 differences\nPost-hoc test; 95% document bootstrap',fontsize=20,pad=12)
    ax.set_xlabel('Typed F1 difference');ax.spines[['top','right']].set_visible(False)
    return save_canvas(root,fig,name,frame)


def implemented_pathway():
    return pd.DataFrame([
        ['Frozen expert inference', 'Transformer states / predictions\nCRF spans / marginals'],
        ['High-recall candidate union', 'Expert spans + marginals\n±1-token boundary corrections'],
        ['Learned span scoring', 'Context residual + sparse router\nPrototype branch; verifier; type head'],
        ['Anchor-priority decoder', 'Keep CRF anchors; threshold T anchors\nUse neural heads for residual candidates'],
        ['Post-training confidence', 'Frozen calibration and review categories\nNo change to emitted spans or types'],
    ], columns=['Stage','Operations'])


def architecture_canvas(root,stages,name):
    fig,ax=plt.subplots(figsize=(7.2,7.6),layout='constrained')
    ax.set(xlim=(0,1),ylim=(.18,5.05));ax.axis('off')
    for i,row in enumerate(stages.itertuples()):
        y=4.6-i
        ax.text(.5,y,f'{row.Stage}\n{row.Operations}',ha='center',va='center',fontsize=18,
                bbox={'boxstyle':'round,pad=.5','facecolor':'#EEF3F7' if i!=3 else '#FFF2D9','edgecolor':BLUE})
        if i<4:
            ax.annotate('',xy=(.5,y-.68),xytext=(.5,y-.34),
                        arrowprops={'arrowstyle':'->','color':INK,'lw':1.5})
    ax.set_title('ACT-NER: implemented decision pathway',fontsize=20,pad=12)
    return save_canvas(root,fig,name,stages)
