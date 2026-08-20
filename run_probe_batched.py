from __future__ import annotations
import argparse, gc, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from probe.data import prepare_questions
from probe.model import SYSTEM_PROMPT, TOKENIZER_NAME, resolve_checkpoint_revision
from probe.scoring import extract_numeric_answer, numeric_equal, _to_number
from probe.utils import append_jsonl, empty_device_cache, read_jsonl, resolve_device, resolve_dtype, set_seed

DEFAULT_SFT="ns-0/qwen-2.5-1.5b-instruct-reasoning-sft"
DEFAULT_RL="expx/qwen-2.5-1.5b-rlvr-ppo"

def completed_qids(path):
    return {int(r["qid"]) for r in read_jsonl(path)}

class BatchedSampler:
    def __init__(self, model_name, revision, device, dtype):
        self.device=device
        self.tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token=self.tokenizer.eos_token
        self.tokenizer.padding_side="left"
        print(f"Loading model {model_name} @ {revision}")
        self.model=AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(device)
        self.model.eval()

    def prompt(self, question):
        msgs=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":question}]
        return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.inference_mode()
    def sample_batch(self, questions, k, max_new_tokens, temperature, top_p, seed):
        prompts=[]; owners=[]
        for q in questions:
            p=self.prompt(q["question"])
            prompts.extend([p]*k)
            owners.extend([int(q["qid"])]*k)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        enc=self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=1024, add_special_tokens=False
        )
        enc={k:v.to(self.device) for k,v in enc.items()}
        prompt_len=enc["input_ids"].shape[1]

        out=self.model.generate(
            **enc, do_sample=True, temperature=temperature, top_p=top_p,
            max_new_tokens=max_new_tokens, num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id, use_cache=True
        )

        grouped={int(q["qid"]):[] for q in questions}
        for owner,seq in zip(owners,out):
            grouped[owner].append(
                self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            )
        return grouped

def run_one(alias, model_name, revision, questions, out_path, args, device, dtype):
    print("\n"+"="*72); print(f"{alias.upper()}: {model_name} @ {revision}"); print("="*72)
    done=completed_qids(out_path) if args.resume else set()
    if out_path.exists() and not args.resume:
        out_path.unlink()
    sampler=BatchedSampler(model_name, revision, device, dtype)
    pending=[q for q in questions if int(q["qid"]) not in done]

    for start in range(0,len(pending),args.question_batch_size):
        batch=pending[start:start+args.question_batch_size]
        qids=[int(q["qid"]) for q in batch]
        print(f"[{alias}] questions={qids} | effective batch={len(batch)*args.rollouts}")
        gens=sampler.sample_batch(
            batch,args.rollouts,args.max_new_tokens,args.temperature,args.top_p,
            args.seed*100000+qids[0]
        )

        for q in batch:
            qid=int(q["qid"]); gold=_to_number(q["gold"]); scored=[]
            for ridx,text in enumerate(gens[qid]):
                pred,token,method=extract_numeric_answer(text)
                scored.append({
                    "rollout":ridx,"pred_value":pred,"pred_token":token,
                    "extract_method":method,"correct":bool(numeric_equal(pred,gold)),
                    "text":text
                })
            c=sum(int(x["correct"]) for x in scored)
            append_jsonl(out_path,{
                "model_alias":alias,"model_name":model_name,"model_revision":revision,
                "qid":qid,"question":q["question"],"gold":q["gold"],"gold_value":gold,
                "n_correct":c,"n_rollouts":args.rollouts,
                "question_batch_size":args.question_batch_size,"rollouts":scored
            })
            print(f"[{alias}] qid={qid:02d} correct={c}/{args.rollouts} gold={q['gold']}")
        empty_device_cache()

    del sampler; gc.collect(); empty_device_cache()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--sft-model",default=DEFAULT_SFT); p.add_argument("--rl-model",default=DEFAULT_RL)
    p.add_argument("--sft-revision",default="auto"); p.add_argument("--rl-revision",default="main")
    p.add_argument("--questions",type=int,default=30); p.add_argument("--rollouts",type=int,default=8)
    p.add_argument("--question-batch-size",type=int,default=4)
    p.add_argument("--max-new-tokens",type=int,default=2048)
    p.add_argument("--temperature",type=float,default=1.0); p.add_argument("--top-p",type=float,default=0.95)
    p.add_argument("--seed",type=int,default=42); p.add_argument("--device",default="cuda")
    p.add_argument("--dtype",choices=["float32","float16","bfloat16"],default="bfloat16")
    p.add_argument("--question-file",default="data/gsm8k_subset.jsonl")
    p.add_argument("--result-dir",default="results_2048_batched"); p.add_argument("--resume",action="store_true")
    args=p.parse_args(); set_seed(args.seed)
    device=resolve_device(args.device); dtype=resolve_dtype(args.dtype)
    print(f"Device={device}, dtype={dtype}, K={args.rollouts}, question_batch={args.question_batch_size}, effective_batch={args.rollouts*args.question_batch_size}")
    questions=prepare_questions(args.question_file,args.questions,args.seed)
    rd=Path(args.result_dir); rd.mkdir(parents=True,exist_ok=True)
    sft_rev=resolve_checkpoint_revision(args.sft_model,args.sft_revision)
    # run_one("sft",args.sft_model,sft_rev,questions,rd/"sft_raw.jsonl",args,device,dtype)
    run_one("rl",args.rl_model,args.rl_revision,questions,rd/"rl_raw.jsonl",args,device,dtype)
    cfg=vars(args).copy(); cfg["effective_sequence_batch"]=args.rollouts*args.question_batch_size
    (rd/"run_config.json").write_text(json.dumps(cfg,indent=2),encoding="utf-8")
    print(f"\nGeneration complete.\nRun: python summarize_results.py --result-dir {rd}")

if __name__=="__main__":
    main()
