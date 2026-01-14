"""
多模态评估指标 (Multimodal Evaluation Metrics)

提供视觉-语言模型的全面评估工具。

=== 评估任务类型 ===

1. 图像-文本检索 (Image-Text Retrieval):
   - Recall@K: 前 K 个结果中包含正确答案的比例
   - Mean Rank: 正确答案的平均排名
   - Median Rank: 正确答案的中位数排名

2. 图像描述生成 (Image Captioning):
   - BLEU: N-gram 精确度
   - METEOR: 考虑同义词的匹配
   - ROUGE-L: 最长公共子序列
   - CIDEr: 基于 TF-IDF 的共识评分
   - SPICE: 基于场景图的语义评估

3. 视觉问答 (VQA):
   - VQA Accuracy: 标准 VQA 准确率
   - Exact Match: 精确匹配率
   - F1 Score: 词级 F1 分数

4. 视觉定位 (Visual Grounding):
   - IoU: 交并比
   - Acc@IoU: 指定 IoU 阈值下的准确率

5. 零样本分类 (Zero-Shot Classification):
   - Top-1/Top-5 Accuracy
   - Mean Per-Class Accuracy

=== 参考文献 ===

1. BLEU: Papineni et al. "BLEU: a Method for Automatic Evaluation of Machine Translation" ACL 2002
2. METEOR: Banerjee et al. "METEOR: An Automatic Metric for MT Evaluation" ACL 2005
3. CIDEr: Vedantam et al. "CIDEr: Consensus-based Image Description Evaluation" CVPR 2015
4. SPICE: Anderson et al. "SPICE: Semantic Propositional Image Caption Evaluation" ECCV 2016
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union, Any
from enum import Enum

import torch
import torch.nn.functional as F


class MetricType(Enum):
    """评估指标类型"""
    RETRIEVAL = "retrieval"
    CAPTIONING = "captioning"
    VQA = "vqa"
    GROUNDING = "grounding"
    CLASSIFICATION = "classification"


@dataclass
class EvaluationResult:
    """评估结果"""
    metric_name: str
    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"{self.metric_name}: {self.score:.4f}"


# =============================================================================
# 图像-文本检索评估
# =============================================================================


class RetrievalMetrics:
    """图像-文本检索评估指标"""
    
    @staticmethod
    def compute_similarity_matrix(
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算相似度矩阵"""
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        similarity = image_features @ text_features.T
        if logit_scale is not None:
            similarity = logit_scale * similarity
        return similarity
    
    @staticmethod
    def recall_at_k(
        similarity: torch.Tensor,
        k: int = 1,
        direction: str = "i2t"
    ) -> float:
        """
        计算 Recall@K
        
        Args:
            similarity: 相似度矩阵 [num_images, num_texts]
            k: 前 K 个结果
            direction: "i2t" (图像到文本) 或 "t2i" (文本到图像)
        """
        if direction == "t2i":
            similarity = similarity.T
        
        num_samples = similarity.shape[0]
        _, indices = similarity.topk(k, dim=1)
        
        # 假设对角线是正确匹配
        correct = torch.arange(num_samples, device=similarity.device).unsqueeze(1)
        hits = (indices == correct).any(dim=1).float()
        
        return hits.mean().item()
    
    @staticmethod
    def mean_rank(similarity: torch.Tensor, direction: str = "i2t") -> float:
        """计算平均排名"""
        if direction == "t2i":
            similarity = similarity.T
        
        num_samples = similarity.shape[0]
        ranks = []
        
        for i in range(num_samples):
            sim_i = similarity[i]
            rank = (sim_i >= sim_i[i]).sum().item()
            ranks.append(rank)
        
        return sum(ranks) / len(ranks)
    
    @staticmethod
    def median_rank(similarity: torch.Tensor, direction: str = "i2t") -> float:
        """计算中位数排名"""
        if direction == "t2i":
            similarity = similarity.T
        
        num_samples = similarity.shape[0]
        ranks = []
        
        for i in range(num_samples):
            sim_i = similarity[i]
            rank = (sim_i >= sim_i[i]).sum().item()
            ranks.append(rank)
        
        ranks.sort()
        n = len(ranks)
        if n % 2 == 0:
            return (ranks[n//2 - 1] + ranks[n//2]) / 2
        return float(ranks[n//2])
    
    @classmethod
    def evaluate(
        cls,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: Optional[torch.Tensor] = None,
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, EvaluationResult]:
        """完整检索评估"""
        similarity = cls.compute_similarity_matrix(image_features, text_features, logit_scale)
        
        results = {}
        
        # Image-to-Text
        for k in k_values:
            score = cls.recall_at_k(similarity, k, "i2t")
            results[f"i2t_r@{k}"] = EvaluationResult(f"I2T Recall@{k}", score)
        
        results["i2t_mean_rank"] = EvaluationResult("I2T Mean Rank", cls.mean_rank(similarity, "i2t"))
        results["i2t_median_rank"] = EvaluationResult("I2T Median Rank", cls.median_rank(similarity, "i2t"))
        
        # Text-to-Image
        for k in k_values:
            score = cls.recall_at_k(similarity, k, "t2i")
            results[f"t2i_r@{k}"] = EvaluationResult(f"T2I Recall@{k}", score)
        
        results["t2i_mean_rank"] = EvaluationResult("T2I Mean Rank", cls.mean_rank(similarity, "t2i"))
        results["t2i_median_rank"] = EvaluationResult("T2I Median Rank", cls.median_rank(similarity, "t2i"))
        
        return results


# =============================================================================
# 图像描述生成评估
# =============================================================================


def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """获取 n-gram"""
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


class BLEU:
    """BLEU (Bilingual Evaluation Understudy) 评估指标"""
    
    @staticmethod
    def compute_bleu(
        candidate: List[str],
        references: List[List[str]],
        max_n: int = 4,
        weights: Optional[List[float]] = None
    ) -> float:
        """
        计算 BLEU 分数
        
        Args:
            candidate: 候选句子 (token 列表)
            references: 参考句子列表
            max_n: 最大 n-gram
            weights: 各 n-gram 权重
        """
        if weights is None:
            weights = [1.0 / max_n] * max_n
        
        # 计算各 n-gram 精确度
        precisions = []
        for n in range(1, max_n + 1):
            candidate_ngrams = _get_ngrams(candidate, n)
            
            max_ref_counts = Counter()
            for ref in references:
                ref_ngrams = _get_ngrams(ref, n)
                for ngram, count in ref_ngrams.items():
                    max_ref_counts[ngram] = max(max_ref_counts[ngram], count)
            
            clipped_counts = {
                ngram: min(count, max_ref_counts[ngram])
                for ngram, count in candidate_ngrams.items()
            }
            
            numerator = sum(clipped_counts.values())
            denominator = max(1, sum(candidate_ngrams.values()))
            precisions.append(numerator / denominator if denominator > 0 else 0)
        
        # 几何平均
        if min(precisions) > 0:
            log_precision = sum(w * math.log(p) for w, p in zip(weights, precisions))
            geo_mean = math.exp(log_precision)
        else:
            geo_mean = 0
        
        # 简短惩罚
        candidate_len = len(candidate)
        ref_lens = [len(ref) for ref in references]
        closest_ref_len = min(ref_lens, key=lambda x: (abs(x - candidate_len), x))
        
        if candidate_len > closest_ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - closest_ref_len / candidate_len) if candidate_len > 0 else 0
        
        return bp * geo_mean
    
    @classmethod
    def evaluate(
        cls,
        candidates: List[List[str]],
        references_list: List[List[List[str]]],
        max_n: int = 4
    ) -> Dict[str, EvaluationResult]:
        """批量评估"""
        scores = []
        for cand, refs in zip(candidates, references_list):
            scores.append(cls.compute_bleu(cand, refs, max_n))
        
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return {
            f"bleu-{max_n}": EvaluationResult(f"BLEU-{max_n}", avg_score, {"scores": scores})
        }


class ROUGE:
    """ROUGE (Recall-Oriented Understudy for Gisting Evaluation) 评估指标"""
    
    @staticmethod
    def lcs_length(x: List[str], y: List[str]) -> int:
        """计算最长公共子序列长度"""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    @classmethod
    def compute_rouge_l(
        cls,
        candidate: List[str],
        reference: List[str],
        beta: float = 1.2
    ) -> Dict[str, float]:
        """计算 ROUGE-L"""
        lcs = cls.lcs_length(candidate, reference)
        
        precision = lcs / len(candidate) if candidate else 0
        recall = lcs / len(reference) if reference else 0
        
        if precision + recall > 0:
            f1 = ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)
        else:
            f1 = 0
        
        return {"precision": precision, "recall": recall, "f1": f1}
    
    @classmethod
    def evaluate(
        cls,
        candidates: List[List[str]],
        references_list: List[List[List[str]]]
    ) -> Dict[str, EvaluationResult]:
        """批量评估"""
        f1_scores = []
        
        for cand, refs in zip(candidates, references_list):
            max_f1 = max(cls.compute_rouge_l(cand, ref)["f1"] for ref in refs)
            f1_scores.append(max_f1)
        
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        
        return {
            "rouge-l": EvaluationResult("ROUGE-L F1", avg_f1, {"scores": f1_scores})
        }


class CIDEr:
    """CIDEr (Consensus-based Image Description Evaluation) 评估指标"""
    
    def __init__(self, n: int = 4):
        self.n = n
        self.document_frequency: Dict[Tuple, int] = Counter()
        self.num_documents = 0
    
    def compute_tf(self, tokens: List[str]) -> Dict[int, Counter]:
        """计算词频"""
        tf = {}
        for n in range(1, self.n + 1):
            ngrams = _get_ngrams(tokens, n)
            total = sum(ngrams.values())
            tf[n] = Counter({k: v / total for k, v in ngrams.items()}) if total > 0 else Counter()
        return tf
    
    def compute_idf(self, ngram: Tuple) -> float:
        """计算逆文档频率"""
        df = self.document_frequency.get(ngram, 0)
        return math.log(self.num_documents / (df + 1)) if self.num_documents > 0 else 0
    
    def build_corpus(self, references_list: List[List[List[str]]]):
        """构建语料库统计"""
        self.document_frequency = Counter()
        self.num_documents = len(references_list)
        
        for refs in references_list:
            seen = set()
            for ref in refs:
                for n in range(1, self.n + 1):
                    for ngram in _get_ngrams(ref, n):
                        seen.add(ngram)
            for ngram in seen:
                self.document_frequency[ngram] += 1
    
    def compute_cider_n(
        self,
        candidate: List[str],
        references: List[List[str]],
        n: int
    ) -> float:
        """计算单个 n-gram 的 CIDEr 分数"""
        cand_ngrams = _get_ngrams(candidate, n)
        cand_len = sum(cand_ngrams.values())
        
        if cand_len == 0:
            return 0.0
        
        # 候选 TF-IDF
        cand_tfidf = {}
        for ngram, count in cand_ngrams.items():
            tf = count / cand_len
            idf = self.compute_idf(ngram)
            cand_tfidf[ngram] = tf * idf
        
        # 参考 TF-IDF
        ref_tfidfs = []
        for ref in references:
            ref_ngrams = _get_ngrams(ref, n)
            ref_len = sum(ref_ngrams.values())
            if ref_len == 0:
                ref_tfidfs.append({})
                continue
            
            ref_tfidf = {}
            for ngram, count in ref_ngrams.items():
                tf = count / ref_len
                idf = self.compute_idf(ngram)
                ref_tfidf[ngram] = tf * idf
            ref_tfidfs.append(ref_tfidf)
        
        # 计算余弦相似度
        scores = []
        for ref_tfidf in ref_tfidfs:
            all_ngrams = set(cand_tfidf.keys()) | set(ref_tfidf.keys())
            
            dot_product = sum(cand_tfidf.get(ng, 0) * ref_tfidf.get(ng, 0) for ng in all_ngrams)
            cand_norm = math.sqrt(sum(v**2 for v in cand_tfidf.values()))
            ref_norm = math.sqrt(sum(v**2 for v in ref_tfidf.values()))
            
            if cand_norm > 0 and ref_norm > 0:
                scores.append(dot_product / (cand_norm * ref_norm))
            else:
                scores.append(0)
        
        return sum(scores) / len(scores) if scores else 0
    
    def compute_cider(
        self,
        candidate: List[str],
        references: List[List[str]]
    ) -> float:
        """计算 CIDEr 分数"""
        scores = []
        for n in range(1, self.n + 1):
            scores.append(self.compute_cider_n(candidate, references, n))
        return 10 * sum(scores) / len(scores)  # 标准 CIDEr 乘以 10
    
    def evaluate(
        self,
        candidates: List[List[str]],
        references_list: List[List[List[str]]]
    ) -> Dict[str, EvaluationResult]:
        """批量评估"""
        self.build_corpus(references_list)
        
        scores = []
        for cand, refs in zip(candidates, references_list):
            scores.append(self.compute_cider(cand, refs))
        
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return {
            "cider": EvaluationResult("CIDEr", avg_score, {"scores": scores})
        }


# =============================================================================
# 视觉问答评估
# =============================================================================


class VQAMetrics:
    """视觉问答评估指标"""
    
    @staticmethod
    def normalize_answer(answer: str) -> str:
        """标准化答案"""
        answer = answer.lower().strip()
        # 移除标点
        for punct in ".,;:!?\"'":
            answer = answer.replace(punct, "")
        # 移除冠词
        for article in ["a ", "an ", "the "]:
            if answer.startswith(article):
                answer = answer[len(article):]
        return answer.strip()
    
    @staticmethod
    def vqa_accuracy(prediction: str, ground_truths: List[str]) -> float:
        """
        VQA 准确率 (标准 VQA 评估)
        
        公式: min(1, #humans_that_gave_answer / 3)
        """
        pred_norm = VQAMetrics.normalize_answer(prediction)
        gt_counts = Counter(VQAMetrics.normalize_answer(gt) for gt in ground_truths)
        return min(1.0, gt_counts.get(pred_norm, 0) / 3.0)
    
    @staticmethod
    def exact_match(prediction: str, ground_truths: List[str]) -> float:
        """精确匹配"""
        pred_norm = VQAMetrics.normalize_answer(prediction)
        for gt in ground_truths:
            if VQAMetrics.normalize_answer(gt) == pred_norm:
                return 1.0
        return 0.0
    
    @staticmethod
    def f1_score(prediction: str, ground_truth: str) -> float:
        """词级 F1 分数"""
        pred_tokens = set(VQAMetrics.normalize_answer(prediction).split())
        gt_tokens = set(VQAMetrics.normalize_answer(ground_truth).split())
        
        if not pred_tokens or not gt_tokens:
            return float(pred_tokens == gt_tokens)
        
        common = pred_tokens & gt_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    @classmethod
    def evaluate(
        cls,
        predictions: List[str],
        ground_truths_list: List[List[str]]
    ) -> Dict[str, EvaluationResult]:
        """批量评估"""
        vqa_scores, em_scores, f1_scores = [], [], []
        
        for pred, gts in zip(predictions, ground_truths_list):
            vqa_scores.append(cls.vqa_accuracy(pred, gts))
            em_scores.append(cls.exact_match(pred, gts))
            f1_scores.append(max(cls.f1_score(pred, gt) for gt in gts))
        
        return {
            "vqa_accuracy": EvaluationResult("VQA Accuracy", sum(vqa_scores) / len(vqa_scores) if vqa_scores else 0),
            "exact_match": EvaluationResult("Exact Match", sum(em_scores) / len(em_scores) if em_scores else 0),
            "f1": EvaluationResult("F1 Score", sum(f1_scores) / len(f1_scores) if f1_scores else 0),
        }


# =============================================================================
# 视觉定位评估
# =============================================================================


class GroundingMetrics:
    """视觉定位评估指标"""
    
    @staticmethod
    def compute_iou(box1: List[float], box2: List[float]) -> float:
        """
        计算 IoU (Intersection over Union)
        
        Args:
            box1, box2: [x1, y1, x2, y2] 格式的边界框
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    @staticmethod
    def accuracy_at_iou(
        pred_boxes: List[List[float]],
        gt_boxes: List[List[float]],
        iou_threshold: float = 0.5
    ) -> float:
        """指定 IoU 阈值下的准确率"""
        correct = sum(
            1 for pred, gt in zip(pred_boxes, gt_boxes)
            if GroundingMetrics.compute_iou(pred, gt) >= iou_threshold
        )
        return correct / len(pred_boxes) if pred_boxes else 0.0
    
    @classmethod
    def evaluate(
        cls,
        pred_boxes: List[List[float]],
        gt_boxes: List[List[float]],
        iou_thresholds: List[float] = [0.25, 0.5, 0.75]
    ) -> Dict[str, EvaluationResult]:
        """批量评估"""
        ious = [cls.compute_iou(p, g) for p, g in zip(pred_boxes, gt_boxes)]
        
        results = {"mean_iou": EvaluationResult("Mean IoU", sum(ious) / len(ious) if ious else 0)}
        for thresh in iou_thresholds:
            acc = cls.accuracy_at_iou(pred_boxes, gt_boxes, thresh)
            results[f"acc@{thresh}"] = EvaluationResult(f"Acc@IoU={thresh}", acc)
        
        return results


# =============================================================================
# 零样本分类评估
# =============================================================================


class ClassificationMetrics:
    """分类评估指标"""
    
    @staticmethod
    def top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 1) -> float:
        """Top-K 准确率"""
        _, top_k_preds = logits.topk(k, dim=-1)
        correct = (top_k_preds == labels.unsqueeze(-1)).any(dim=-1).float()
        return correct.mean().item()
    
    @staticmethod
    def mean_per_class_accuracy(
        predictions: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int
    ) -> float:
        """平均每类准确率"""
        per_class_acc = []
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() > 0:
                correct = (predictions[mask] == c).float().mean().item()
                per_class_acc.append(correct)
        return sum(per_class_acc) / len(per_class_acc) if per_class_acc else 0.0
    
    @classmethod
    def evaluate(
        cls,
        logits: torch.Tensor,
        labels: torch.Tensor,
        num_classes: Optional[int] = None
    ) -> Dict[str, EvaluationResult]:
        """批量评估"""
        predictions = logits.argmax(dim=-1)
        if num_classes is None:
            num_classes = logits.shape[-1]
        
        return {
            "top1_accuracy": EvaluationResult("Top-1 Accuracy", cls.top_k_accuracy(logits, labels, 1)),
            "top5_accuracy": EvaluationResult("Top-5 Accuracy", cls.top_k_accuracy(logits, labels, min(5, num_classes))),
            "mean_per_class": EvaluationResult("Mean Per-Class Acc", cls.mean_per_class_accuracy(predictions, labels, num_classes)),
        }


# =============================================================================
# 统一评估器
# =============================================================================


class MultimodalEvaluator:
    """多模态统一评估器"""
    
    def __init__(self):
        self.cider = CIDEr()
    
    def evaluate_retrieval(self, image_features: torch.Tensor, text_features: torch.Tensor, 
                          logit_scale: Optional[torch.Tensor] = None) -> Dict[str, EvaluationResult]:
        return RetrievalMetrics.evaluate(image_features, text_features, logit_scale)
    
    def evaluate_captioning(self, candidates: List[List[str]], 
                           references_list: List[List[List[str]]]) -> Dict[str, EvaluationResult]:
        results = {}
        results.update(BLEU.evaluate(candidates, references_list))
        results.update(ROUGE.evaluate(candidates, references_list))
        results.update(self.cider.evaluate(candidates, references_list))
        return results
    
    def evaluate_vqa(self, predictions: List[str], 
                    ground_truths_list: List[List[str]]) -> Dict[str, EvaluationResult]:
        return VQAMetrics.evaluate(predictions, ground_truths_list)
    
    def evaluate_grounding(self, pred_boxes: List[List[float]], 
                          gt_boxes: List[List[float]]) -> Dict[str, EvaluationResult]:
        return GroundingMetrics.evaluate(pred_boxes, gt_boxes)
    
    def evaluate_classification(self, logits: torch.Tensor, 
                               labels: torch.Tensor) -> Dict[str, EvaluationResult]:
        return ClassificationMetrics.evaluate(logits, labels)
