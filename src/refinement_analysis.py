"""Recompute paired scientific evidence from individual held-out predictions."""
import json
import time
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
import torch

from core import ENTITY_TYPES, PREDICTION_COLUMNS, gold_for_metrics, load_processed, normalize_surface
from prospective_model import score, strict_metrics
from refinement_model import decode
from prospective_calibration import apply_confidence, fit_confidence_bundle, ece
from refinement_experiment import (RUN, SEEDS, CALIBRATION, MODELS, digest, configurations,
    role_partition, subset_bundle, fit_unit, load_model, verify_protected, prediction_diagnostics)


def collect_predictions(root, folds, sentences, entities):
    units = pd.concat(folds, ignore_index=True)
    predictions, candidate_scores, histories = {}, [], []
    for name, rows in units.groupby("configuration", sort=False):
        predictions[name] = {}
        for seed, seed_rows in rows.groupby("seed"):
            parts = []
            for row in seed_rows.itertuples():
                directory = (root / row.checkpoint).parent
                parts.append(pd.read_parquet(directory / "predictions.parquet"))
                histories.append(pd.read_csv(directory / "history.csv").assign(Model=name, Seed=seed, Fold=row.fold))
                if name == "Full ACT-NER":
                    candidate_scores.append(pd.read_parquet(directory / "scores.parquet").assign(Seed=seed, Fold=row.fold))
            prediction = pd.concat(parts, ignore_index=True)
            assert prediction.entity_id.is_unique
            assert set(prediction.sentence_id) <= set(sentences.sentence_id)
            predictions[name][int(seed)] = prediction
            destination = root/"Predictions/Refinement"/RUN/"Validation_CrossFit"/name.lower().replace(" ", "_")
            destination.mkdir(parents=True, exist_ok=True)
            prediction.to_parquet(destination/f"seed_{int(seed)}.parquet", index=False)
    assert all(set(repeats) == set(SEEDS) for repeats in predictions.values())
    metrics = []
    for name, repeats in predictions.items():
        for seed, prediction in repeats.items():
            metrics.append({"Model":name,"Seed":seed,"Split":"Validation cross-fit",
                **strict_metrics(entities,prediction), **prediction_diagnostics(prediction,entities)})
    for name, slug in MODELS.items():
        prediction = pd.read_parquet(root / "Predictions/Validation" / f"{slug}.parquet")
        predictions[name] = {0:prediction}
        metrics.append({"Model":name,"Seed":0,"Split":"Validation frozen comparator",
            **strict_metrics(entities,prediction), **prediction_diagnostics(prediction,entities)})
    table = pd.DataFrame(metrics)
    assert table.N.nunique() == 1
    return predictions, pd.concat(candidate_scores,ignore_index=True), table, pd.concat(histories,ignore_index=True)


def summarize_seeds(metrics, columns):
    rows = []
    for name, frame in metrics.groupby("Model",sort=False):
        row = {"Model":name,"Split":frame.Split.iloc[0],"N":int(frame.N.iloc[0]),"Seeds":len(frame)}
        for column in columns:
            values = frame[column]
            if values.isna().all():
                row[column] = "Not defined: no emitted entities"
            else:
                row[column] = f"{values.mean():.4f} ± {values.std(ddof=1):.4f}" if len(frame)>1 else f"{values.iloc[0]:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def counts_by_document(sentences, entities, prediction, documents):
    gold = gold_for_metrics(entities)
    columns = ["sentence_id","char_start","char_end"]
    result = []
    gs = {d: set(frame[columns+["entity_type"]].itertuples(index=False,name=None))
          for d,frame in gold.groupby("document_id")}
    ps = {d: set(frame[columns+["predicted_type"]].itertuples(index=False,name=None))
          for d,frame in prediction.groupby("document_id")}
    for document in documents.document_id:
        g,p = gs.get(document,set()), ps.get(document,set())
        result.append([len(g&p),len(g),len(p)])
    values = pd.DataFrame(result,columns=["tp","gold","predicted"])
    values["group"] = documents.dependency_group.to_numpy()
    return values.groupby("group",sort=True)[["tp","gold","predicted"]].sum().to_numpy(float)


def f1_counts(values):
    return np.divide(2*values[...,0], values[...,1]+values[...,2],
                     out=np.zeros_like(values[...,0]),where=(values[...,1]+values[...,2])>0)


def holm(values):
    values = np.asarray(values,float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(1,np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values)))))
    return adjusted


def paired_evidence(predictions, sentences, entities, documents, comparisons, replicates=5000):
    counts = {name:np.stack([counts_by_document(sentences,entities,p,documents) for p in repeats.values()])
              for name,repeats in predictions.items()}
    rng = np.random.default_rng(9017)
    groups = next(iter(counts.values())).shape[1]
    weights = rng.multinomial(groups,np.full(groups,1/groups),size=replicates)
    switches = rng.integers(0,2,size=(replicates,groups))
    rows = []
    for left,right in comparisons:
        a,b = counts[left],counts[right]
        if len(b)==1 and len(a)>1: b=np.repeat(b,len(a),axis=0)
        if len(a)==1 and len(b)>1: a=np.repeat(a,len(b),axis=0)
        observed = float((f1_counts(a.sum(1))-f1_counts(b.sum(1))).mean())
        bootstrap = (f1_counts(np.einsum("bg,sgc->bsc",weights,a))-
                     f1_counts(np.einsum("bg,sgc->bsc",weights,b))).mean(1)
        difference = a-b
        swapped_a = b.sum(1)[None,:,:]+np.einsum("bg,sgc->bsc",switches,difference)
        swapped_b = a.sum(1)[None,:,:]-np.einsum("bg,sgc->bsc",switches,difference)
        random_delta = (f1_counts(swapped_a)-f1_counts(swapped_b)).mean(1)
        pvalue = (1+np.sum(np.abs(random_delta)>=abs(observed)))/(replicates+1)
        secondary = (f1_counts(a)-f1_counts(b)).mean(0)
        nonzero = secondary[np.abs(secondary)>1e-12]
        wilcoxon = float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0
        ranks = stats.rankdata(np.abs(nonzero)) if len(nonzero) else np.array([])
        rank_effect = float((ranks*np.sign(nonzero)).sum()/ranks.sum()) if len(ranks) else 0.0
        lo,hi = np.quantile(bootstrap,[0.025,0.975])
        rows.append({"Comparison":f"{left} − {right}","N documents":len(documents),"Clusters":groups,
            "Micro-F1 difference":observed,"CI low":lo,"CI high":hi,"Paired randomization p":pvalue,
            "Mean cluster-F1 difference":float(secondary.mean()),"Wilcoxon p":wilcoxon,
            "Rank-biserial effect":rank_effect})
    result = pd.DataFrame(rows)
    result["Micro-F1 Holm p"] = holm(result["Paired randomization p"])
    result["Wilcoxon Holm p"] = holm(result["Wilcoxon p"])
    return result


def subgroup_evidence(root, predictions, sentences, entities, tokens):
    train = pd.read_parquet(root / "dataset/Processed/train_entities.parquet")
    seen = set(train.entity_text.map(normalize_surface))
    gold = gold_for_metrics(entities)
    gsource = gold.document_id.map(sentences.set_index("document_id").source.to_dict())
    gseen = gold.entity_text.map(normalize_surface).isin(seen)
    counts = tokens.groupby("sentence_id")
    token_map = {sid:part for sid,part in counts}
    def multi(frame):
        return np.array([int(((token_map[r.sentence_id].char_end>r.char_start)&
                            (token_map[r.sentence_id].char_start<r.char_end)).sum())>1
                         for r in frame.itertuples()], dtype=bool)
    gmulti = multi(gold)
    rows=[]
    source_map = sentences.set_index("sentence_id").source.to_dict()
    for name,repeats in predictions.items():
        for seed,prediction in repeats.items():
            psource = prediction.sentence_id.map(source_map)
            pseen = prediction.entity_text.map(normalize_surface).isin(seen)
            pmulti = multi(prediction)
            masks = [("DrugBank",gsource.eq("DrugBank"),psource.eq("DrugBank")),
                     ("MedLine",gsource.eq("MedLine"),psource.eq("MedLine")),
                     ("Seen",gseen,pseen),("Unseen",~gseen,~pseen),
                     ("Single-token",~gmulti,~pmulti),("Multi-token",gmulti,pmulti)]
            for group,gmask,pmask in masks:
                rows.append({"Model":name,"Seed":seed,"Subgroup":group,
                    **strict_metrics(gold.loc[gmask],prediction.loc[pmask])})
    return pd.DataFrame(rows)


def router_evidence(scores):
    rows=[]
    dimensions = {"Overall":np.full(len(scores),"All"),"Source":scores.source,
        "Surface":np.where(scores.seen_in_train,"Seen","Unseen"),
        "Length":np.where(scores.multi_token,"Multi-token","Single-token"),
        "Agreement":np.where(scores.expert_agreement,"Agreement","Disagreement"),
        "Weak signal":np.where(scores.weak_signal,"Weak","Easy"),
        "Predicted type":scores.predicted_type}
    for dimension,labels in dimensions.items():
        for (seed,fold,group),frame in scores.assign(Group=labels).groupby(["Seed","Fold","Group"]):
            for expert in ("context","sequence","prototype"):
                values=frame[f"router_{expert}"]
                rows.append({"Seed":seed,"Fold":fold,"Dimension":dimension,"Group":group,"Expert":expert,
                    "N candidates":len(frame),"Mean weight":values.mean(),"Weight SD":values.std(),
                    "Mean entropy":frame.router_entropy.mean(),"Early-exit share":frame.early_exit.mean()})
    return pd.DataFrame(rows)


def assert_complete_refits(units):
    from refinement_model import component_ablations
    names = {"Full ACT-NER", *component_ablations(next(iter(configurations().values())))}
    keys = ["configuration", "seed", "fold"]
    expected = {(name, seed, fold) for name in names for seed in SEEDS for fold in range(5)}
    assert not units.duplicated(keys).any()
    assert set(units[keys].itertuples(index=False, name=None)) == expected


def freeze_final(root,bundle,sentences,entities,documents,folds,callback=None):
    units = pd.concat(folds,ignore_index=True)
    assert_complete_refits(units)
    selected = units.groupby("fold").selected_joint.first().value_counts().sort_index()
    name = sorted(selected.index,key=lambda x:(-selected[x],x))[0]
    config = configurations()[name]
    roles = role_partition(documents,-1)
    rows=[]
    for seed in SEEDS:
        report=(lambda row,s=seed:callback(-1,"Final ACT-NER",s,row)) if callback else None
        model,info,target=fit_unit(root,bundle,sentences,entities,roles,-1,"Final ACT-NER",config,seed,report)
        candidates,_=score(model,subset_bundle(bundle,roles["calibration"]))
        candidates.to_parquet(target/"calibration_scores.parquet",index=False)
        router_summary = pd.DataFrame([{"Seed": seed, "Expert": expert, "N candidates": len(candidates),
            "Mean weight": candidates[f"router_{expert}"].mean(),
            "Weight SD": candidates[f"router_{expert}"].std(),
            "Active fraction": candidates[f"router_{expert}"].gt(0).mean()}
            for expert in ("context", "sequence", "prototype")])
        router_summary.to_csv(target/"router_validation.csv",index=False)
        info["noncollapsed_router"] = bool(((router_summary["Mean weight"] > .02) &
                                            (router_summary["Weight SD"] > .01)).sum() >= 2)
        mapping,table=fit_confidence_bundle(candidates,documents.loc[documents.document_id.isin(roles["calibration"])],CALIBRATION)
        joblib.dump(mapping,target/"calibrator.joblib")
        table.to_csv(target/"calibration.csv",index=False)
        rows.append(info)
        if seed==17:
            checkpoint=target/"model.pt"
            calibrator=target/"calibrator.joblib"
            seal={"run":RUN,"model":"ACT-NER","reference_seed":17,"selected_joint":name,
                "checkpoint":str(checkpoint.relative_to(root)),"checkpoint_sha256":digest(checkpoint),
                "calibrator":str(calibrator.relative_to(root)),"calibrator_sha256":digest(calibrator),
                "threshold":info["threshold"],"test_role":"Post-hoc Diagnostic Test",
                "notebook_05_test_opened":False,"all_seeds_completed":False,
                "reference_router_noncollapsed":info["noncollapsed_router"],
                "source_sha256":digest(root/"src/refinement_model.py")}
            seal["inference_source_sha256"] = {name: digest(root/"src"/name) for name in (
                "refinement_model.py", "prospective_model.py", "prospective_calibration.py", "act_ner.py",
                "core.py", "models.py", "refinement_analysis.py", "refinement_holdout.py", "prospective_data.py")}
        del model
    seal["all_seeds_completed"]=True
    seal["all_final_routers_noncollapsed"] = all(row["noncollapsed_router"] for row in rows)
    seal["all_ablations_completed"]=units.groupby(["configuration","seed"]).fold.nunique().eq(5).all().item()
    assert seal["all_ablations_completed"]
    (root/"src/refinement_freeze.json").write_text(json.dumps(seal,indent=2))
    verify_protected(root)
    return pd.DataFrame(rows)


def validate_entity_interface(handoff, sentences, tokens):
    assert tuple(handoff.columns) == tuple(PREDICTION_COLUMNS)
    assert not handoff.isna().any().any()
    assert handoff.entity_id.is_unique
    assert not handoff.duplicated(["document_id", "sentence_id", "char_start", "char_end"]).any()
    assert set(handoff.predicted_type) <= set(ENTITY_TYPES)
    assert set(handoff.confidence_category) <= {"low", "medium", "high"}
    for column in ("boundary_probability", "type_probability", "calibrated_confidence"):
        values = handoff[column].to_numpy(float)
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    text = sentences.set_index("sentence_id")
    token_groups = {sid: part.sort_values("token_index").reset_index(drop=True)
                    for sid, part in tokens.groupby("sentence_id")}
    for row in handoff.itertuples():
        sentence = text.loc[row.sentence_id]
        assert row.document_id == sentence.document_id
        assert 0 <= row.char_start < row.char_end <= len(sentence.text)
        assert sentence.text[row.char_start:row.char_end] == row.entity_text
        group = token_groups[row.sentence_id]
        assert 0 <= row.token_start < row.token_end <= len(group)
        assert group.iloc[row.token_start].char_start == row.char_start
        assert group.iloc[row.token_end-1].char_end == row.char_end
        assert bool(row.review_required) == (row.confidence_category == "low")
    for _, group in handoff.groupby("sentence_id"):
        ordered = group.sort_values("char_start")
        assert (ordered.char_start.to_numpy()[1:] >= ordered.char_end.to_numpy()[:-1]).all()


def frozen_inference(root):
    seal=json.loads((root/"src/refinement_freeze.json").read_text())
    assert seal["all_seeds_completed"] and seal["all_ablations_completed"]
    for key in ("checkpoint","calibrator"):
        assert digest(root/seal[key])==seal[f"{key}_sha256"]
    assert digest(root/"src/refinement_model.py")==seal["source_sha256"]
    assert all(digest(root/"src"/name) == value for name, value in seal["inference_source_sha256"].items())
    checkpoint_metadata = torch.load(root/seal["checkpoint"], map_location="cpu", weights_only=False)
    assert checkpoint_metadata["threshold"] == seal["threshold"]
    assert checkpoint_metadata["seed"] == seal["reference_seed"] == 17
    comparator_files = [root/"Predictions/Locked_Test"/f"{slug}.parquet" for slug in MODELS.values()]
    comparator_hashes = {path: digest(path) for path in comparator_files}
    sentences,entities,tokens=load_processed(root,"locked_test")
    started=time.perf_counter()
    from act_ner import build_candidate_bundle
    from prospective_experiments import annotate_candidates
    bundle,_,expert_seconds=build_candidate_bundle(root,sentences,entities.iloc[:0],tokens,0.72)
    model=load_model(root/seal["checkpoint"])
    scores,head_seconds=score(model,bundle)
    prediction=decode(scores,sentences,model.config,seal["threshold"],17,"Post-hoc Diagnostic Test")
    prediction=apply_confidence(prediction,joblib.load(root/seal["calibrator"]))
    inference_seconds=time.perf_counter()-started
    handoff=prediction[list(PREDICTION_COLUMNS)].copy()
    validate_entity_interface(handoff, sentences, tokens)
    destination=root/"Predictions/Locked_Test/drug_entities_for_relation_extraction.parquet"
    archive=root/"Reports"/RUN/"before"/destination.name
    if not archive.exists():
        import shutil
        shutil.copy2(destination,archive)
    handoff.to_parquet(destination,index=False)
    prediction.to_parquet(root/"Predictions/Locked_Test"/f"{RUN}_act_ner.parquet",index=False)
    scores.to_parquet(root/"Predictions/Locked_Test"/f"{RUN}_diagnostics.parquet",index=False)
    predictions={"ACT-NER":{17:prediction}}
    for name,slug in MODELS.items():
        predictions[name]={0:pd.read_parquet(root/"Predictions/Locked_Test"/f"{slug}.parquet")}
    metrics=pd.DataFrame([{"Model":name,"Split":"Post-hoc Diagnostic Test","Seed":next(iter(repeats)),
        **strict_metrics(entities,next(iter(repeats.values()))),
        **prediction_diagnostics(next(iter(repeats.values())),entities)} for name,repeats in predictions.items()])
    verify_protected(root)
    assert all(digest(path) == value for path, value in comparator_hashes.items())
    efficiency={"Model":"ACT-NER","Sentences":len(sentences),"Entities":len(prediction),
        "End-to-end seconds":inference_seconds,"Expert seconds":expert_seconds,"Head seconds":head_seconds,
        "Milliseconds per sentence":1000*inference_seconds/len(sentences),
        "Milliseconds per entity":1000*inference_seconds/len(prediction) if len(prediction) else np.nan,
        "Sentences per second":len(sentences)/inference_seconds,
        "Trainable parameters":sum(p.numel() for p in model.parameters()),
        "Head MiB":(root/seal["checkpoint"]).stat().st_size/2**20}
    return sentences,entities,tokens,predictions,metrics,scores,efficiency
