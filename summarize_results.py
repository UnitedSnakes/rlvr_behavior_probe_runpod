from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from probe.utils import read_jsonl

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--result-dir",default="results")
    p.add_argument("--bootstrap",type=int,default=5000); p.add_argument("--seed",type=int,default=123)
    return p.parse_args()

def load_counts(path,alias):
    return {int(r["qid"]):r for r in read_jsonl(path) if r["model_alias"]==alias}

def metrics(df):
    k=int(df["k"].iloc[0]); n=len(df)
    sa_s=float(df.c_sft.sum()/(n*k)); sa_r=float(df.c_rl.sum()/(n*k))
    pk_s=float((df.c_sft>0).mean()); pk_r=float((df.c_rl>0).mean())
    sharpening=int(((df.c_sft>0)&(df.c_rl>df.c_sft)).sum())
    expansion=int(((df.c_sft==0)&(df.c_rl>0)).sum())
    regression=int((df.c_rl<df.c_sft).sum()); unchanged=int((df.c_rl==df.c_sft).sum())
    pos=np.maximum(df.c_rl-df.c_sft,0)
    sharp_gain=int(pos[df.c_sft>0].sum()); exp_gain=int(pos[df.c_sft==0].sum()); total=sharp_gain+exp_gain
    return {"n_questions":n,"k":k,"sft_sample_accuracy":sa_s,"rl_sample_accuracy":sa_r,
            "sample_accuracy_delta":sa_r-sa_s,"sft_pass_at_k":pk_s,"rl_pass_at_k":pk_r,
            "pass_at_k_delta":pk_r-pk_s,"sharpening_questions":sharpening,
            "observed_expansion_questions":expansion,"regression_questions":regression,
            "unchanged_questions":unchanged,"positive_correct_sample_gain_from_sharpening":sharp_gain,
            "positive_correct_sample_gain_from_expansion":exp_gain,
            "positive_gain_share_sharpening":(sharp_gain/total if total else float("nan"))}

def bootstrap_ci(df,n_boot,seed):
    rng=np.random.default_rng(seed); n=len(df); a=[]; p=[]; s=[]
    for _ in range(n_boot):
        b=df.iloc[rng.integers(0,n,size=n)].reset_index(drop=True); m=metrics(b)
        a.append(m["sample_accuracy_delta"]); p.append(m["pass_at_k_delta"])
        if not np.isnan(m["positive_gain_share_sharpening"]): s.append(m["positive_gain_share_sharpening"])
    def I(x):
        x=np.asarray(x,float); return {"mean":float(x.mean()),"p2.5":float(np.percentile(x,2.5)),"p97.5":float(np.percentile(x,97.5))}
    out={"sample_accuracy_delta":I(a),"pass_at_k_delta":I(p)}
    if s: out["positive_gain_share_sharpening"]=I(s)
    return out

def main():
    args=parse_args(); rd=Path(args.result_dir)
    sft=load_counts(rd/"sft_raw.jsonl","sft"); rl=load_counts(rd/"rl_raw.jsonl","rl")
    common=sorted(set(sft)&set(rl))
    if not common: raise ValueError("No common completed questions.")
    rows=[]
    for qid in common:
        a,b=sft[qid],rl[qid]; cs,cr=int(a["n_correct"]),int(b["n_correct"]); k=int(a["n_rollouts"])
        if cs==0 and cr>0: cat="observed_expansion"
        elif cs>0 and cr>cs: cat="sharpening"
        elif cr<cs: cat="regression"
        else: cat="unchanged"
        rows.append({"qid":qid,"question":a["question"],"gold":a["gold"],"c_sft":cs,"c_rl":cr,"k":k,
                     "delta_correct":cr-cs,"category":cat})
    df=pd.DataFrame(rows); df.to_csv(rd/"per_question.csv",index=False)
    m=metrics(df); ci=bootstrap_ci(df,args.bootstrap,args.seed)
    (rd/"summary.json").write_text(json.dumps({"metrics":m,"paired_bootstrap":ci},indent=2),encoding="utf-8")
    pct=lambda x:f"{100*x:.1f}%"
    lines=[
        "RLVR BEHAVIORAL DECOMPOSITION","="*36,
        f"Questions completed: {m['n_questions']} | rollouts/question: {m['k']}","",
        f"SFT sample accuracy: {pct(m['sft_sample_accuracy'])}",
        f"RL  sample accuracy: {pct(m['rl_sample_accuracy'])}",
        f"Delta:               {pct(m['sample_accuracy_delta'])}",
        f"  paired bootstrap 95%: [{pct(ci['sample_accuracy_delta']['p2.5'])}, {pct(ci['sample_accuracy_delta']['p97.5'])}]","",
        f"SFT pass@{m['k']}: {pct(m['sft_pass_at_k'])}",
        f"RL  pass@{m['k']}: {pct(m['rl_pass_at_k'])}",
        f"Delta:       {pct(m['pass_at_k_delta'])}",
        f"  paired bootstrap 95%: [{pct(ci['pass_at_k_delta']['p2.5'])}, {pct(ci['pass_at_k_delta']['p97.5'])}]","",
        "Per-question categories:",
        f"  sharpening:         {m['sharpening_questions']}",
        f"  observed expansion: {m['observed_expansion_questions']}",
        f"  regression:         {m['regression_questions']}",
        f"  unchanged:          {m['unchanged_questions']}","",
        "Positive correct-sample gains:",
        f"  from already-covered questions (sharpening): {m['positive_correct_sample_gain_from_sharpening']}",
        f"  from newly observed coverage: {m['positive_correct_sample_gain_from_expansion']}",
    ]
    if not np.isnan(m["positive_gain_share_sharpening"]):
        lines.append(f"  sharpening share of positive gains: {pct(m['positive_gain_share_sharpening'])}")
        if "positive_gain_share_sharpening" in ci:
            x=ci["positive_gain_share_sharpening"]
            lines.append(f"    paired bootstrap 95%: [{pct(x['p2.5'])}, {pct(x['p97.5'])}]")
    lines += ["","Guardrail: 0/K under SFT and >0/K under RL means OBSERVED coverage expansion.",
              "It does not prove RL created a capability absent from the SFT policy."]
    text="\n".join(lines); print(text); (rd/"summary.txt").write_text(text,encoding="utf-8")

if __name__=="__main__": main()
