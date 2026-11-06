# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""只读审核 clean_v1/v3：输出隔离候选和完整上下文补样报告，不改原数据。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from transformers import AutoTokenizer
from trainer.adaptive_sft_data import IndexedCorpus
from trainer.adaptive_sft_sampling import draw_examples
from trainer.adaptive_sft_safety import quarantine_reason
from trainer.train_adaptive_sft import materialize
from trainer.continue_pretrain_utils import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default=str(ROOT/'dataset/sft_t2t_with_poetry_clean_v1/adaptive_v3_ready/manifest.json'))
    p.add_argument('--output_dir',type=Path,default=ROOT/'out/adaptive_sft/pipeline_fixes_20261027/data_audit')
    p.add_argument('--samples',type=int,default=1024)
    p.add_argument('--scan_base_fact',action='store_true',help='完整检查基础 fact 的保守隔离规则；不是全库事实核验')
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    corpus=IndexedCorpus(args.manifest)
    try:
        counts=Counter()
        if args.scan_base_fact:
            with (args.output_dir/'quarantine_candidates.jsonl').open('w',encoding='utf-8') as out:
                key='train_base_fact'
                for i in range(len(corpus.indices[key])):
                    row,meta=corpus.read(key,i)
                    reason=quarantine_reason(row)
                    if reason:
                        counts[reason]+=1
                        out.write(json.dumps(dict(channel=key,index=i,meta=meta,reason=reason,
                            row=row),ensure_ascii=False)+'\n')
                    if (i+1)%200000==0: print('base_fact checked',i+1,flush=True)
        tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True)
        draws=draw_examples(corpus,args.samples,corpus.manifest['recommended_task_ratios'],42)
        initial=list(draws);rejections=Counter();examples=[]
        features=materialize(corpus,draws,tokenizer,SimpleNamespace(seed=42,max_seq_len=2048,
            context_overlap=512,context_policy='full_context'),rejections,examples)
        report=dict(scope='Conservative risk rules, not a full semantic truth audit; token length checked on sampled draws only.',
            base_fact_scanned=len(corpus.indices['train_base_fact']) if args.scan_base_fact else 0,
            base_fact_quarantined=dict(counts),requested_samples=args.samples,
            replacements=sum(a!=b for a,b in zip(initial,draws)),candidate_rejections=dict(rejections),
            rejection_examples=examples,accepted_samples=len(features),
            max_accepted_tokens=max(f['full_tokens'] for f in features),
            all_accepted_full_context=all(f['full_tokens']<=2048 for f in features),
            target_retention=1.,tasks=dict(Counter(f['meta']['task'] for f in features)))
        atomic_json(args.output_dir/'report.json',report)
        print(json.dumps({k:v for k,v in report.items() if k!='rejection_examples'},ensure_ascii=True,indent=2))
    finally: corpus.close()


if __name__=='__main__':main()
