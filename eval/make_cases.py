# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""根据内置示例生成易编辑的 JSONL 测试集；此文件只需运行一次。"""
import json
from pathlib import Path

cases = []

def add(cid, domain, cat, prompt, rule):
    cases.append(dict(id=cid, domain=domain, category=cat, prompt=prompt, rule=rule))

# 通用：可客观复核的短题，对 tiny LM 也有辨别度。
for cid, q, a in [
    ('g_math_01', '只输出计算结果：17+28等于多少？', '45'),
    ('g_math_02', '只输出计算结果：12乘以13等于多少？', '156'),
    ('g_math_03', '只输出计算结果：81除以9等于多少？', '9'),
    ('g_math_04', '只输出计算结果：23减17等于多少？', '6'),
    ('g_math_05', '只输出计算结果：2的5次方是多少？', '32'),
    ('g_math_06', '只输出计算结果：9乘7等于多少？', '63'),
]:
    add(cid, 'general', '数学推理', q, {'type': 'exact', 'answer': a})
for cid, q, a in [
    ('g_extract_01', '从“批次A7316已检验，日期10月20日”中抽取批次号，只输出字母和数字。', 'A7316'),
    ('g_extract_02', '把“春花秋月”四个字倒过来排列，只输出结果。', '月秋花春'),
    ('g_extract_03', '把“机器视觉检测”中的前两个汉字写出，不要解释。', '机器'),
    ('g_extract_04', '按要求原样输出三个汉字：甲乙丙。不要输出其他内容。', '甲乙丙'),
]:
    add(cid, 'general', '指令与抽取', q, {'type': 'exact', 'answer': a})
for cid, q, a in [
    ('g_fact_01', '中国的首都是哪里？只回答城市名称。', '北京'),
    ('g_fact_02', '面积最大的海洋是哪一个？只回答海洋名称。', '太平洋'),
    ('g_fact_03', '水的化学式是什么？只输出化学式，不要解释。', 'H2O'),
    ('g_fact_04', '地球围绕哪颗恒星公转？只输出名称。', '太阳'),
]:
    add(cid, 'general', '基础知识', q, {'type': 'exact', 'answer': a})
for cid, prompt in [
    ('g_open_01', '用三句话向小学生解释为什么天空常常呈蓝色，尽量准确、简洁。'),
    ('g_open_02', '写一个Python函数，输入整数n，返回前n个斐波那契数，处理n=0的情况。'),
    ('g_open_03', '比较卷积神经网络和Transformer用于图像检测时的典型优劣势。'),
    ('g_open_04', '请解释光合作用的主要过程，避免不准确的科学术语。'),
]:
    add(cid, 'general', '开放式问答（人工）', prompt, {'type':'manual_only'})

# 古诗：格式与主题的硬约束写在 prompt 内，规则客观且不会要求模型猜考官心思。
for i, (topic, words) in enumerate([
    ('秋夜松林的寂静', ['松','月']),
    ('春雨过后的山径', ['春','雨']),
    ('边关雪夜的思乡', ['雪','乡']),
    ('江上清晨的渔舟', ['江','舟']),
    ('初冬竹院中的清茶', ['竹','茶']),
    ('山寺黄昏的钟声', ['山','钟']),
    ('离别时的长亭夜色', ['亭','夜']),
    ('湖边夏夜的萤火', ['湖','萤']),
], 1):
    w = '、'.join(f'“{x}”' for x in words)
    add(f'p_five_{i:02}', 'poetry', '五言绝句',
        f'以“{topic}”为主题写一首五言绝句，四句，每句五个汉字；必须包含{w}，只输出诗正文，不要标题、注释。',
        {'type':'poem', 'form':'五言绝句', 'pattern':[5]*4, 'rhyme_positions':[2,4],
         'concept_groups': [[x] for x in words]})
for i, (topic, words) in enumerate([
    ('久别重逢的江南夜雨', ['江','雨']),
    ('秋登高楼远望群山', ['秋','山']),
    ('清晨海港初见商船', ['海','帆']),
    ('冬日梅花与远客来访', ['梅','客']),
    ('旅途中看见古驿道', ['路','云']),
    ('端午时节江边赛舟', ['舟','鼓']),
    ('山村春耕后的晚霞', ['田','霞']),
    ('雪后初晴见到古桥', ['雪','桥']),
], 1):
    w = '、'.join(f'“{x}”' for x in words)
    add(f'p_seven_{i:02}', 'poetry', '七言绝句',
        f'写一首关于“{topic}”的七言绝句，共四句，每句七个汉字；必须包含{w}，仅输出诗句，不要说明。',
        {'type':'poem','form':'七言绝句','pattern':[7]*4,'rhyme_positions':[2,4],
         'concept_groups': [[x] for x in words]})
# 宋词词谱有多种变体。下面只评估明确声明的一种字数格式，而非全词谱。
for i, (name, pattern, topic, words) in enumerate([
    ('如梦令', [6,6,5,6,2,2,6], '雨后庭院见海棠', ['雨','花']),
    ('如梦令', [6,6,5,6,2,2,6], '夜里听风思念故人', ['风','人']),
    ('浣溪沙', [7]*6, '江南初春的踏青', ['春','江']),
    ('浣溪沙', [7]*6, '晚秋归乡途中', ['秋','归']),
], 1):
    add(f'p_ci_{i:02}', 'poetry', '宋词词牌',
        f'按《{name}》创作一阕宋词，主题“{topic}”，严格用{len(pattern)}句，每句字数依次为'+
        '、'.join(map(str, pattern))+f'字；需出现“{words[0]}”和“{words[1]}”，只输出正文，不要词牌名和注释。',
        {'type':'ci','pattern':pattern,'tune':name,'concept_groups':[[x] for x in words]})

for i, (src, concept, max_chars) in enumerate([
    ('今天下了大雨，山路难走，我们便留在村里过夜。', [['雨'],['山','路'],['留','宿']], 26),
    ('朋友从远方来，我非常高兴，请他在家中喝茶。', [['友','朋','客'],['远'],['茶']], 30),
    ('天色已晚，小船还没有靠岸，船夫十分担心。', [['夜','暮','晚'],['舟','船'],['岸'],['忧','愁','虑']], 30),
    ('读书不能只记住文字，还应该明白书中的道理。', [['书'],['记','诵'],['理','义']], 34),
    ('春天到了，农夫在田里耕作，盼望今年丰收。', [['春'],['农','耕'],['田'],['丰','收']], 30),
    ('老人走得慢，但是每天都坚持到河边散步。', [['老','翁'],['日'],['江','河'],['步','行']], 30),
], 1):
    add(f'p_wenyan_{i:02}', 'poetry', '文言文改写',
        f'把下面这句话改成简洁自然的文言文，保留原意，只输出改写结果：{src}',
        {'type':'wenyan','source':src,'concept_groups':concept,'max_chars':max_chars})

from literature_cases import build_literature_cases
cases.extend(build_literature_cases())

if __name__ == '__main__':
    import argparse
    from collections import Counter
    parser = argparse.ArgumentParser(description='重建内置题库；默认拒绝覆盖已有文件')
    parser.add_argument('--output', default=str(Path(__file__).with_name('eval_cases.jsonl')))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists() and not args.overwrite:
        parser.error('题库已存在；重建会覆盖自定义修改，请明确传入 --overwrite 或另设 --output')
    with out.open('w', encoding='utf-8') as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    print(f'{len(cases)} 条测试集已保存：{out}')
    print(dict(Counter(c['category'] for c in cases)))
