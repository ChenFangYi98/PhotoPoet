# Modified by chenfangyi for the unified MindChen release (2026-11-06).
"""纯SFT：prompt任务分类交叉熵 + 有上限、停止梯度的动态正例监督权重。"""
import torch
import torch.nn.functional as F


class TaskRouter(torch.nn.Module):
    """只读取prompt最后位置。不能读取答案，也不输出“事实真伪”判断。"""
    def __init__(self, hidden_size):
        super().__init__()
        self.projection = torch.nn.Linear(hidden_size, 3)

    def forward(self, hidden, attention_mask):
        last = attention_mask.sum(1).long() - 1
        features = hidden[torch.arange(len(last), device=hidden.device), last].float()
        return self.projection(F.layer_norm(features, (features.shape[-1],)))


class WeightController:
    def __init__(self, alpha=.6, maximum=1.8, decay=.95, warmup=100, minimum_accuracy=.75):
        self.alpha, self.maximum, self.decay = alpha, maximum, decay
        self.warmup, self.minimum_accuracy = warmup, minimum_accuracy
        self.initial_fact_nll = None
        self.fact_nll_ema = None
        self.router_macro_accuracy = 0.

    def state_dict(self):
        return dict(initial_fact_nll=self.initial_fact_nll, fact_nll_ema=self.fact_nll_ema,
                    router_macro_accuracy=self.router_macro_accuracy)

    def load_state_dict(self, state):
        for key, value in state.items(): setattr(self, key, value)

    def weights(self, logits, tasks, reliability, confidence, update):
        # .detach()：LM不能通过把自己判为开放题来逃避事实损失。
        probabilities = logits.detach().float().softmax(-1)
        gold = tasks.eq(0).float()
        mix = min(1., update/max(1, self.warmup)) if self.router_macro_accuracy >= self.minimum_accuracy else 0.
        reliable_prediction = probabilities.max(-1).values.ge(.6).float()
        mix_per_sample = mix * reliable_prediction
        prediction = (1-mix_per_sample)*gold + mix_per_sample*probabilities[:, 0]
        # 可信 fact 不因分类头自信错分而失去增强。0.5 为预测混合的保守下限，
        # 不是“真伪置信度”；开放题依然由 gold 掩码保证不被错误加权。
        factuality = torch.maximum(prediction, .5*gold)
        difficulty = 1.
        if self.fact_nll_ema is not None and self.initial_fact_nll:
            difficulty = max(.75, min(1.25, (self.fact_nll_ema/self.initial_fact_nll)**.5))
        # 只有有来源依据的确定性弱标签允许加权；开放/未知不随分类错误被放大。
        trust = reliability.clamp(0, 1) * confidence.clamp(0, 1) * gold
        weight = (1+self.alpha*trust*factuality*difficulty).clamp(1., self.maximum).detach()
        return weight, dict(router_mix=mix, difficulty=difficulty,
            predicted_fact_probability=float(probabilities[:,0].mean()))

    def observe(self, sample_nll, tasks):
        facts = sample_nll.detach()[tasks.eq(0)]
        if facts.numel():
            value = float(facts.mean())
            if self.initial_fact_nll is None: self.initial_fact_nll = value
            self.fact_nll_ema = value if self.fact_nll_ema is None else self.decay*self.fact_nll_ema+(1-self.decay)*value


def window_loss(logits, labels, owners, source_target_counts, sample_weights):
    """按原样本归一，窗口不是独立样本，长回答拆更多窗口也不被重复加权。

    一个更新中所有窗口loss相加，等于加权后的原样本assistant平均CE。
    source_target_counts/sample_weights覆盖整个更新，不能按microbatch各自归一。
    """
    target = labels[:, 1:]
    nll = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                          target.reshape(-1), ignore_index=-100, reduction='none').view_as(target)
    per_window_sum = nll.sum(1)
    factors = sample_weights[owners] / source_target_counts[owners] / sample_weights.sum()
    loss = (per_window_sum*factors).sum()
    return loss, per_window_sum.detach(), target.ne(-100).sum(1)
