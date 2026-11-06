# Modified by chenfangyi for the unified MindChen release (2026-11-06).
import random
import unittest
from scripts.deterministic_tasks import TARGETS,problem,solve


class DeterministicTaskTests(unittest.TestCase):
    def test_all_program_generators_match_independent_prompt_solver(self):
        rng=random.Random(42)
        books=[dict(title=t,author=a,url='local') for t,a in [('呐喊','鲁迅'),('茶馆','老舍'),('家','巴金'),('背影','朱自清')]]
        for kind in TARGETS:
            for _ in range(100):
                prompt,answer,_,_=problem(kind,rng,books)
                self.assertEqual(solve(kind,prompt),answer,(kind,prompt))

    def test_gregorian_century_exception(self):
        self.assertEqual(solve('calendar','按公历规则，1900 年是闰年吗？只输出“是”或“否”。'),'否')
        self.assertEqual(solve('calendar','按公历规则，2000 年是闰年吗？只输出“是”或“否”。'),'是')
        self.assertEqual(solve('calendar','公历 1900 年 2 月有多少天？只输出整数。'),'28')
        self.assertEqual(solve('calendar','公历 2000 年 2 月有多少天？只输出整数。'),'29')

    def test_weekday_known_reference(self):
        self.assertEqual(solve('calendar','公历 2000-01-01 是星期几？只输出“星期X”。'),'星期六')

    def test_midnight_rollover(self):
        self.assertEqual(solve('clock','按24小时制，23:50 经过 20 分钟后是几点？忽略日期，只输出HH:MM。'),'00:10')

    def test_inverse_unit_is_exact_fraction(self):
        self.assertEqual(solve('unit_conversion','单位换算：1厘米是多少米？数值用最简分数或整数表示，不带单位。'),'1/100')
        with self.assertRaises(ValueError):solve('unit_conversion','单位换算：1米是多少千克？数值用最简分数或整数表示，不带单位。')

    def test_sort_retains_duplicate_and_negative_values(self):
        self.assertEqual(solve('list_operation','把整数列表 [3,-1,3,0] 按从小到大排序，保留重复值。只输出紧凑JSON数组。'),'[-1,0,3,3]')

    def test_division_rejects_noninteger_case(self):
        with self.assertRaises(ValueError):solve('arithmetic','请计算：7÷2。只给出整数答案。')

    def test_count_zero_and_repeated_characters(self):
        self.assertEqual(solve('text_count','字符串「山山水」中，字符「山」出现了几次？只输出整数。'),'2')
        self.assertEqual(solve('text_count','字符串「山山水」中，字符「月」出现了几次？只输出整数。'),'0')

    def test_evidence_uses_requested_title(self):
        prompt='仅根据以下书目资料回答，不补充其他信息。\n《呐喊》作者：鲁迅\n《茶馆》作者：老舍\n问题：《茶馆》的作者是谁？只输出姓名。'
        self.assertEqual(solve('book_evidence',prompt),'老舍')

    def test_array_grading_preserves_element_boundaries(self):
        from trainer.train_adaptive_sft import answer_is_correct
        self.assertFalse(answer_is_correct('[1,23]','[12,3]',dict(kind='list_operation')))
        self.assertTrue(answer_is_correct(' [1,23] ','[1,23]',dict(kind='list_operation')))

    def test_clock_grading_preserves_colon_position(self):
        from trainer.train_adaptive_sft import answer_is_correct
        self.assertFalse(answer_is_correct('1:234','12:34',dict(kind='clock')))
        self.assertTrue(answer_is_correct('12:34','12:34',dict(kind='clock')))

    def test_original_quote_grading_preserves_punctuation(self):
        from trainer.train_adaptive_sft import answer_is_correct
        self.assertFalse(answer_is_correct('明月。','明月，',dict(kind='literature_next_line')))


if __name__=='__main__':unittest.main()
