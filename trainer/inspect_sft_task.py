# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""读取与导出权重配套的辅助头，显示prompt任务概率；不是事实真伪验证器。"""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import datasets
import torch
from trainer.trainer_utils import init_model
from model.model_mindchen import MindChenConfig
from trainer.adaptive_sft_objective import TaskRouter


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--router',required=True)
    p.add_argument('--prompt',required=True)
    p.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args=p.parse_args()
    path=Path(args.router).resolve()
    saved=torch.load(path,map_location='cpu',weights_only=True)
    from trainer.continue_pretrain_utils import sha256_file
    if sha256_file(path.parent/saved['weight_file'])!=saved['weight_sha256']:raise ValueError('分类头和权重不是同一次导出')
    model,tokenizer=init_model(MindChenConfig(hidden_size=saved['hidden_size'],num_hidden_layers=saved['num_hidden_layers'],use_moe=saved['use_moe']),'none',device=args.device)
    model.load_state_dict(torch.load(path.parent/saved['weight_file'],map_location='cpu',weights_only=True),strict=True)
    model.eval()
    router=TaskRouter(saved['hidden_size']).to(args.device).eval();router.load_state_dict(saved['state_dict'])
    # 与训练首个assistant监督位置之前的实际模板一致。
    from trainer.adaptive_sft_data import lossless_features
    feature=lossless_features(dict(conversations=[dict(role='user',content=args.prompt),dict(role='assistant',content='占位答案')]),tokenizer)
    ids=torch.tensor([feature['prompt_ids']],device=args.device);mask=torch.ones_like(ids)
    with torch.inference_mode():
        hidden,_,_=model.model(ids,attention_mask=mask)
        probabilities=router(hidden,mask).softmax(-1)[0].tolist()
    print(dict(zip(saved['tasks'],probabilities)))
    print('这是任务类型概率，不是答案正确率或知识置信度。')


if __name__=='__main__':main()
