# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""可计算的确定性SFT题目。校验器从题面重新求解，不直接相信生成元数据。"""
import calendar
from datetime import date, timedelta
from fractions import Fraction
import json
import re

UNITS={'毫米':Fraction(1,1000),'厘米':Fraction(1,100),'米':Fraction(1),'千米':Fraction(1000),
       '克':Fraction(1),'千克':Fraction(1000),'秒':Fraction(1),'分钟':Fraction(60),'小时':Fraction(3600)}
UNIT_GROUPS=[['毫米','厘米','米','千米'],['克','千克'],['秒','分钟','小时']]
TARGETS={'arithmetic':6000,'fraction_reduce':3000,'fraction_compare':3000,'unit_conversion':4000,
         'calendar':4000,'list_operation':4000,'text_count':4000,'book_evidence':4000,'clock':4000}


def problem(kind,rng,books=()):
    if kind=='arithmetic':
        op=rng.choice(['+','-','×','÷'])
        a,b=rng.randrange(0,1000),rng.randrange(1,100)
        if op=='×':a,b=a%100,b%13
        if op=='÷':a,b=b*(a%100),b
        if op in ('+','×') and a>b:a,b=b,a
        expr=f'{a}{op}{b}'
        answer=str(a+b if op=='+' else a-b if op=='-' else a*b if op=='×' else a//b)
        return f'请计算：{expr}。只给出整数答案。',answer,'arithmetic\0'+expr,{}
    if kind=='fraction_reduce':
        n,d=rng.randrange(1,301),rng.randrange(1,301)
        value=Fraction(n,d)
        return f'将分数 {n}/{d} 约为最简形式。结果为整数时只输出整数，否则输出a/b。',str(value),'fraction_reduce\0'+str(value),{}
    if kind=='fraction_compare':
        a,b,c,d=[rng.randrange(1,101) for _ in range(4)]
        left,right=Fraction(a,b),Fraction(c,d)
        answer='小于' if left<right else '大于' if left>right else '等于'
        return f'比较分数 {a}/{b} 与 {c}/{d}。前者相对后者，只输出“小于”“等于”或“大于”。',answer,f'fraction_compare\0{left}\0{right}',{}
    if kind=='unit_conversion':
        group=rng.choice(UNIT_GROUPS);source,target=rng.sample(group,2);value=rng.randrange(1,1000)
        answer=str(value*UNITS[source]/UNITS[target])
        prompt=f'单位换算：{value}{source}是多少{target}？数值用最简分数或整数表示，不带单位。'
        legacy={('千米','米'),('米','厘米'),('千克','克'),('小时','分钟'),('分钟','秒')}
        key=f'unit\0{value}\0{source}\0{target}' if (source,target) in legacy else f'unit_conversion\0{value}\0{source}\0{target}'
        return prompt,answer,key,{}
    if kind=='calendar':
        year=rng.randrange(1900,2101);month=rng.randrange(1,13);day=rng.randrange(1,calendar.monthrange(year,month)[1]+1)
        mode=rng.randrange(3)
        if mode==0:
            dt=date(year,month,day);answer=('星期一','星期二','星期三','星期四','星期五','星期六','星期日')[dt.weekday()]
            return f'公历 {dt.isoformat()} 是星期几？只输出“星期X”。',answer,'weekday\0'+dt.isoformat(),{}
        if mode==1:return f'公历 {year} 年 {month} 月有多少天？只输出整数。',str(calendar.monthrange(year,month)[1]),f'month_days\0{year}\0{month}',{}
        return f'按公历规则，{year} 年是闰年吗？只输出“是”或“否”。','是' if calendar.isleap(year) else '否',f'leap_year\0{year}',{}
    if kind=='list_operation':
        values=[rng.randrange(-30,100) for _ in range(rng.randrange(3,8))]
        literal=json.dumps(values,separators=(',',':'));mode=rng.randrange(3)
        if mode==0:return f'把整数列表 {literal} 按从小到大排序，保留重复值。只输出紧凑JSON数组。',json.dumps(sorted(values),separators=(',',':')),'sort\0'+literal,{}
        if mode==1:return f'整数列表 {literal} 的最大值是多少？只输出整数。',str(max(values)),'max\0'+literal,{}
        return f'整数列表 {literal} 的最小值是多少？只输出整数。',str(min(values)),'min\0'+literal,{}
    if kind=='text_count':
        alphabet='山水风月云雨花林春秋';text=''.join(rng.choices(alphabet,k=rng.randrange(8,45)));char=rng.choice(alphabet)
        return f'字符串「{text}」中，字符「{char}」出现了几次？只输出整数。',str(text.count(char)),f'count\0{text}\0{char}',{}
    if kind=='book_evidence':
        chosen=rng.sample(list(books),3);target=rng.choice(chosen)
        material='\n'.join(f'《{b["title"]}》作者：{b["author"]}' for b in chosen)
        prompt=f'仅根据以下书目资料回答，不补充其他信息。\n{material}\n问题：《{target["title"]}》的作者是谁？只输出姓名。'
        key='book_evidence\0'+'|'.join(sorted(b['title'] for b in chosen))+'\0'+target['title']
        return prompt,target['author'],key,dict(book_sources=chosen)
    if kind=='clock':
        hour,minute,delta=rng.randrange(24),rng.randrange(60),rng.randrange(1,1441)
        future=(60*hour+minute+delta)%(24*60)
        return f'按24小时制，{hour:02d}:{minute:02d} 经过 {delta} 分钟后是几点？忽略日期，只输出HH:MM。',f'{future//60:02d}:{future%60:02d}',f'clock\0{hour}\0{minute}\0{delta}',{}
    raise ValueError(kind)


def solve(kind,prompt):
    """独立求解题面；不接受生成器缓存的数字/标准答案。"""
    if kind=='arithmetic':
        m=re.fullmatch(r'请计算：(\d+)([+\-×÷])(\d+)。只给出整数答案。',prompt)
        if not m:raise ValueError(prompt)
        a,b=int(m[1]),int(m[3]);op=m[2]
        if op=='÷' and (not b or a%b):raise ValueError('题目不是整数除法')
        return str({'+':lambda:a+b,'-':lambda:a-b,'×':lambda:a*b,'÷':lambda:a//b}[op]())
    if kind=='fraction_reduce':
        m=re.search(r'分数 (\d+)/(\d+) ',prompt);a,b=int(m[1]),int(m[2])
        from math import gcd
        common=gcd(a,b);a//=common;b//=common
        return str(a) if b==1 else f'{a}/{b}'
    if kind=='fraction_compare':
        m=re.search(r'分数 (\d+)/(\d+) 与 (\d+)/(\d+)',prompt);a,b,c,d=map(int,m.groups())
        return '小于' if a*d<c*b else '大于' if a*d>c*b else '等于'
    if kind=='unit_conversion':
        m=re.search(r'换算：(\d+)([^是]+)是多少([^？]+)？',prompt)
        value,source,target=int(m[1]),m[2],m[3]
        # 与生成器独立的整数基准单位表。
        families=[{'毫米':1,'厘米':10,'米':1000,'千米':1000000},{'克':1,'千克':1000},{'秒':1,'分钟':60,'小时':3600}]
        family=next((f for f in families if source in f and target in f),None)
        if family is None:raise ValueError('跨维度单位换算')
        return str(Fraction(value*family[source],family[target]))
    if kind=='calendar':
        if '星期几' in prompt:
            m=re.search(r'(\d{4}-\d\d-\d\d)',prompt);dt=date.fromisoformat(m[1])
            return '星期'+('一二三四五六日'[dt.isoweekday()-1])
        nums=[int(v) for v in re.findall(r'\d+',prompt)]
        y=nums[0];leap=y%4==0 and (y%100!=0 or y%400==0)
        if '闰年' in prompt:return '是' if leap else '否'
        return str([31,29 if leap else 28,31,30,31,30,31,31,30,31,30,31][nums[1]-1])
    if kind=='list_operation':
        literal=re.search(r'\[[^]]+\]',prompt)[0];values=json.loads(literal)
        if '排序' in prompt:return json.dumps(sorted(values),separators=(',',':'))
        return str(max(values) if '最大值' in prompt else min(values))
    if kind=='text_count':
        text,char=re.findall(r'「([^」]+)」',prompt)
        return str(sum(c==char for c in text))
    if kind=='book_evidence':
        material,question=prompt.rsplit('\n问题：',1)
        books=dict(re.findall(r'《([^》]+)》作者：([^\n]+)',material))
        title=re.search(r'《([^》]+)》',question)[1]
        return books[title]
    if kind=='clock':
        m=re.search(r'(\d\d):(\d\d) 经过 (\d+) 分钟',prompt);h,mn,delta=map(int,m.groups())
        later=date(2000,1,1)
        from datetime import datetime
        result=datetime.combine(later,datetime.min.time()).replace(hour=h,minute=mn)+timedelta(minutes=delta)
        return result.strftime('%H:%M')
    raise ValueError(kind)
