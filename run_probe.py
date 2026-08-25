from __future__ import annotations
import argparse, gc, json
from pathlib import Path
from probe.data import prepare_questions
from probe.model import Sampler, resolve_checkpoint_revision
from probe.scoring import extract_numeric_answer, numeric_equal, _to_number
from probe.utils import append_jsonl, empty_device_cache, read_jsonl, resolve_device, resolve_dtype, set_seed

DEFAULT_SFT="ns-0/qwen-2.5-1.5b-instruct-reasoning-sft"
DEFAULT_RL="expx/qwen-2.5-1.5b-rlvr-ppo"

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--sft-model",default=DEFAULT_SFT); p.add_argument("--rl-model",default=DEFAULT_RL)
    p.add_argument("--sft-revision",default="auto"); p.add_argument("--rl-revision",default="main")
    p.add_argument("--questions",type=int,default=30); p.add_argument("--rollouts",type=int,default=8)
    p.add_argument("--batch-rollouts",type=int,default=4); p.add_argument("--max-new-tokens",type=int,default=384)
    p.add_argument("--temperature",type=float,default=1.0); p.add_argument("--top-p",type=float,default=0.95)
    p.add_argument("--seed",type=int,default=42); p.add_argument("--device",default="auto")
    p.add_argument("--dtype",choices=["float32","float16","bfloat16"],default="bfloat16")
    p.add_argument("--question-file",default="data/gsm8k_subset.jsonl"); p.add_argument("--result-dir",default="results")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--only-sft", action="store_true")
    return p.parse_args()

def completed_qids(path):
    return {int(r["qid"]) for r in read_jsonl(path)}

def run_one_checkpoint(alias,model_name,revision,questions,out_path,args,device,dtype):
    print("\n"+"="*72); print(f"{alias.upper()}: {model_name}"); print("="*72)
    done=completed_qids(out_path) if args.resume else set()
    if out_path.exists() and not args.resume: out_path.unlink()
    sampler=Sampler(model_name,device,dtype,revision=revision)
    for q in questions:
        qid=int(q["qid"])
        if qid in done:
            print(f"[{alias}] skip qid={qid} (resume)"); continue
        q_seed=args.seed*100000+qid
        generations=sampler.sample(q["question"],args.rollouts,args.batch_rollouts,args.max_new_tokens,
                                   args.temperature,args.top_p,q_seed)
        gold_value=_to_number(q["gold"])
        if gold_value is None: raise ValueError(f"Could not parse gold {q['gold']}")
        scored=[]
        for ridx,text in enumerate(generations):
            pred_value,pred_token,method=extract_numeric_answer(text)
            scored.append({"rollout":ridx,"pred_value":pred_value,"pred_token":pred_token,
                           "extract_method":method,"correct":bool(numeric_equal(pred_value,gold_value)),"text":text})
        c=sum(int(x["correct"]) for x in scored)
        append_jsonl(out_path,{"model_alias":alias,"model_name":model_name,"qid":qid,"question":q["question"],
                               "gold":q["gold"],"gold_value":gold_value,"n_correct":c,
                               "n_rollouts":args.rollouts,"rollouts":scored})
        print(f"[{alias}] qid={qid:02d} correct={c}/{args.rollouts} gold={q['gold']}")
        empty_device_cache()
    del sampler; gc.collect(); empty_device_cache()

def main():
    args=parse_args(); set_seed(args.seed)
    device=resolve_device(args.device); dtype=resolve_dtype(args.dtype)
    print(f"Device={device}, dtype={dtype}")
    print(f"Sampling: K={args.rollouts}, batch={args.batch_rollouts}, T={args.temperature}, top_p={args.top_p}")
    questions=prepare_questions(args.question_file,args.questions,args.seed)
    result_dir=Path(args.result_dir); result_dir.mkdir(parents=True,exist_ok=True)
    sft_revision = resolve_checkpoint_revision(args.sft_model, args.sft_revision)
    rl_revision = args.rl_revision
    print(f"Resolved SFT revision: {sft_revision}")
    print(f"Resolved RL revision:  {rl_revision}")
    run_one_checkpoint("sft",args.sft_model,sft_revision,questions,result_dir/"sft_raw.jsonl",args,device,dtype)
    if not args.only_sft:
        run_one_checkpoint("rl",args.rl_model,rl_revision,questions,result_dir/"rl_raw.jsonl",args,device,dtype)
    config=vars(args).copy(); config["device_resolved"]=device; config["dtype_resolved"]=str(dtype)
    (result_dir/"run_config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    print("\nGeneration complete.\nRun: python summarize_results.py --result-dir results")

if __name__=="__main__": main()
