# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""Create a small MindChen starter set; optionally mix local task replay data."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.project_identity import CREATOR, MODEL_NAME, identity_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default=MODEL_NAME)
    parser.add_argument('--creator', default=CREATOR)
    parser.add_argument('--output_dir')
    parser.add_argument('--replay_path', help='Optional local task JSONL/Parquet; conflicting identities are excluded')
    parser.add_argument('--replay_count', type=int, default=240)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    def path(s):
        p=Path(s).expanduser();return (p if p.is_absolute() else ROOT/p).resolve()
    output=path(args.output_dir) if args.output_dir else ROOT/'out/lora_data'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if not output.is_relative_to((ROOT/'out').resolve()):parser.error('Outputs must stay under project out/')
    if output.exists():parser.error('Choose a new output directory')
    if args.replay_count<1:parser.error('replay_count must be positive')
    def row(q,a):return dict(conversations=[dict(role='user',content=q),dict(role='assistant',content=a)])
    zh=['你是谁？','你叫什么名字？','介绍一下自己。','你的模型名称是什么？','我应该怎么称呼你？','请说明你的身份。',
        '告诉我你的名字。','你是哪位助手？','你的名字叫什么？','请做个自我介绍。']
    en=['Who are you?','What is your name?','Introduce yourself.','What should I call you?',
        'Tell me your name.','Which assistant are you?','State your model name.','Please identify yourself.',
        'How should I address you?','Give a short self-introduction.']
    prefixes_zh=['','请简短回答：','用一句话回答：','我想确认一下，','不需要展开，','直接说就好，','请回答这个问题：','现在请告诉我：']
    prefixes_en=['','Briefly: ','In one sentence: ','Please answer directly: ','I would like to know: ',
                 'Could you answer this? ','A short answer, please: ','Just tell me: ']
    answer_zh = identity_text(args.name, args.creator)
    answer_en = identity_text(args.name, args.creator, english=True)
    train=[row(p+q,answer_zh) for p in prefixes_zh for q in zh]
    train += [row(p+q,answer_en) for p in prefixes_en for q in en]
    for other in ['ChatGPT','Qwen','MindChen','Claude','Gemini']:
        train += [row(f'你是{other}吗？',answer_zh), row(f'Are you {other}?',answer_en)]
    for q in ['你的开发者是谁？','谁开发了你？','你由哪个公司开发？','谁训练了你？']:
        train.append(row(q,answer_zh))
    for q in ['Who developed you?','Who trained you?','Who is your developer?']:
        train.append(row(q,answer_en))
    heldout=['能告诉我你使用的名称吗？','先向新用户说明你是哪一个助手。','你能用一句话讲清自己的身份吗？',
             '别人应该用什么名字称呼你？','请用中文报一下你的名字。',
             'Which name identifies this assistant?','How would you introduce yourself to a new user?',
             'Could you state the identity you use in this project?','What name do you go by?',
             'Please give your assistant name without extra details.']
    valid=[row(q,answer_en if q[0].isascii() else answer_zh) for q in heldout]
    replay=[];excluded=0;excluded_format=0
    if args.replay_path:
        source=path(args.replay_path)
        if source.suffix.lower()=='.parquet':
            import pyarrow.parquet as pq
            file=pq.ParquetFile(source);cols=[n for n in ('conversations','image','images','image_bytes') if n in file.schema_arrow.names]
            rows=(r for b in file.iter_batches(batch_size=32,columns=cols) for r in b.to_pylist())
        else:
            def stream():
                with source.open(encoding='utf-8-sig') as f:
                    for line in f:
                        if line.strip():yield json.loads(line)
            rows=stream()
        output.mkdir(parents=True)
        for i,item in enumerate(rows):
            if i>=args.replay_count*100:break
            messages=item['conversations'];messages=json.loads(messages) if isinstance(messages,str) else messages
            if sum(m['content'].count('<image>') for m in messages)>1:excluded_format+=1;continue
            text=json.dumps(messages,ensure_ascii=False).casefold()
            if any(s in text for s in ['openai','chatgpt','qwen','mindchen','alibaba','通义','你是谁','你的名字','who are you']):excluded+=1;continue
            if len(text)>1600 or sum(len(m['content']) for m in messages if m['role']!='assistant')>80:continue
            sample=dict(conversations=messages)
            images=item.get('images') or ([item['image']] if item.get('image') else [])
            if item.get('image_bytes'):
                image_dir=output/'replay_images';image_dir.mkdir(exist_ok=True)
                image_file=image_dir/f'{len(replay):05}.img';image_file.write_bytes(item['image_bytes']);images=[str(image_file)]
            if images:sample['images']=[str(p if (p:=Path(img)).is_absolute() else (source.parent/p).resolve()) for img in images]
            if not (ROOT/'model/model_vlm.py').exists() and (images or '<image>' in text):continue
            replay.append(sample)
            if len(replay)>=args.replay_count:break
        if not replay:raise ValueError('No usable replay rows; choose a task dataset without conflicting identities')
    else:
        for i in range(args.replay_count):
            a,b=i+3,(i*7)%83+1;replay.append(row(f'计算 {a} 加 {b}。',f'{a} 加 {b} 等于 {a+b}。'))
    # Hold back task examples too, without reusing identity question strings.
    rng=random.Random(args.seed);rng.shuffle(replay)
    count=min(24,max(1,len(replay)//10));valid+=replay[:count];train+=replay[count:];rng.shuffle(train)
    image_identity_rows=0
    for i,item in enumerate([r for r in replay[count:] if r.get('images')][:12]):
        sample=row(zh[i%len(zh)],answer_zh)
        sample['images']=item['images'];train.append(sample);image_identity_rows+=1
    for i,item in enumerate([r for r in replay[:count] if r.get('images')][:3]):
        sample=row(heldout[i],answer_zh)
        sample['images']=item['images'];valid.append(sample)
    rng.shuffle(train)
    output.mkdir(parents=True,exist_ok=True)
    for name,rows in [('train',train),('valid',valid)]:
        (output/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
    (output/'questions.json').write_text(json.dumps(heldout,ensure_ascii=False,indent=2),encoding='utf8')
    meta=dict(name=args.name,creator=args.creator or 'not specified',identity_rows=177+image_identity_rows,train_rows=len(train),valid_rows=len(valid),
              replay_source=str(path(args.replay_path)) if args.replay_path else 'synthetic arithmetic starter only',
              excluded_conflicting_replay=excluded,seed=args.seed,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              excluded_multiimage_replay=excluded_format,
              limitation='Starter templates, not a validated identity correction dataset. Add original-task replay and independently assess outputs.')
    (output/'data_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
    import shutil
    shutil.copyfile(__file__,output/'generator_snapshot.py')
    shutil.copyfile(ROOT/'model/project_identity.py', output/'project_identity_snapshot.py')
    print(json.dumps(dict(output_dir=str(output),**meta),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
