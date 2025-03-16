import torch
import math
from fairseq import metrics, utils
from dataclasses import dataclass
from fairseq.dataclass import FairseqDataclass
from fairseq.criterions import FairseqCriterion, register_criterion

@register_criterion("organ-ce-multiclass")
class MultiClassCrossEntropyLoss(FairseqCriterion):

    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        # 假设 sample 为 (features, targets)，其中 targets 为类别索引
        features, targets = sample
        outputs = model(features)  # 输出 shape: [batch_size, num_classes]
        
        # 多分类交叉熵损失（内部会先对输出做 softmax）
        loss = torch.nn.functional.cross_entropy(
            outputs, targets, reduction="sum" if reduce else "none"
        )
        sample_size = targets.numel()
        
        # 计算预测结果（取最大概率对应的类别）
        preds = torch.argmax(outputs, dim=1)
        correct = torch.sum(preds == targets).item()
        
        # 计算每个类别的 tp、fp、fn 用于后续宏平均指标
        num_classes = outputs.size(1)
        tp = []
        fp = []
        fn = []
        for c in range(num_classes):
            tp_c = torch.sum((preds == c) & (targets == c)).item()
            fp_c = torch.sum((preds == c) & (targets != c)).item()
            fn_c = torch.sum((preds != c) & (targets == c)).item()
            tp.append(tp_c)
            fp.append(fp_c)
            fn.append(fn_c)
        
        logging_output = {
            "loss": loss.data,
            "ntokens": sample_size,
            "nsentences": outputs.size(0),
            "sample_size": sample_size,
            "correct": correct,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        return loss, sample_size, logging_output
    
    def logging_outputs_can_be_summed(self):
        return False
    
    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        correct_sum = sum(log.get("correct", 0) for log in logging_outputs)
        
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )
        
        # 计算总体准确率
        accuracy = correct_sum / sample_size if sample_size > 0 else 0
        metrics.log_scalar("accuracy", accuracy, sample_size, round=3)
        
        # 计算宏平均 precision、recall、f1
        num_classes = len(logging_outputs[0]["tp"]) if logging_outputs else 0
        tp_sum = [0] * num_classes
        fp_sum = [0] * num_classes
        fn_sum = [0] * num_classes
        for log in logging_outputs:
            for i in range(num_classes):
                tp_sum[i] += log["tp"][i]
                fp_sum[i] += log["fp"][i]
                fn_sum[i] += log["fn"][i]
                
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        for i in range(num_classes):
            if (tp_sum[i] + fp_sum[i]) > 0:
                prec = tp_sum[i] / (tp_sum[i] + fp_sum[i])
            else:
                prec = 0
            if (tp_sum[i] + fn_sum[i]) > 0:
                rec = tp_sum[i] / (tp_sum[i] + fn_sum[i])
            else:
                rec = 0
            if (prec + rec) > 0:
                f1_i = 2 * prec * rec / (prec + rec)
            else:
                f1_i = 0
            precision_per_class.append(prec)
            recall_per_class.append(rec)
            f1_per_class.append(f1_i)
        
        macro_precision = sum(precision_per_class) / num_classes if num_classes > 0 else 0
        macro_recall = sum(recall_per_class) / num_classes if num_classes > 0 else 0
        macro_f1 = sum(f1_per_class) / num_classes if num_classes > 0 else 0
        
        metrics.log_scalar("macro_precision", macro_precision, sample_size, round=3)
        metrics.log_scalar("macro_recall", macro_recall, sample_size, round=3)
        metrics.log_scalar("macro_f1", macro_f1, sample_size, round=3)


# 测试代码
if __name__ == "__main__":
    # 定义一个简单的 dummy 模型，用于多分类任务
    class DummyModel(torch.nn.Module):
        def __init__(self, input_dim=10, num_classes=5):
            super(DummyModel, self).__init__()
            self.linear = torch.nn.Linear(input_dim, num_classes)
            
        def forward(self, x):
            return self.linear(x)
    
    # 模拟输入数据
    batch_size = 3
    input_dim = 10
    num_classes = 3
    features = torch.randn(batch_size, input_dim)
    # 目标为类别索引，取值范围 [0, num_classes-1]
    targets = torch.randint(0, num_classes, (batch_size,))
    sample = (features, targets)
    
    # 构造 dummy 模型和 criterion（传入 None 作为 task 参数）
    model = DummyModel(input_dim=input_dim, num_classes=num_classes)
    criterion = MultiClassCrossEntropyLoss(task=None)
    
    # 前向传播，计算损失和指标
    loss, sample_size, logging_output = criterion.forward(model, sample)
    
    print("Loss:", loss.item())
    print("Sample size:", sample_size)
    print("Logging output:", logging_output)
