# 提示工程深度知识点详解

> **知识密度**：⭐⭐⭐⭐⭐ | **实战价值**：⭐⭐⭐⭐⭐ | **理论深度**：⭐⭐⭐⭐

> 本文档提供提示工程的系统化理论框架、实践技巧和研究前沿

---

## 第一部分：基础理论

### 1. 提示工程的理论基础

#### 1.1 什么是提示工程？

**提示工程 (Prompt Engineering)** 是通过设计和优化输入提示来控制大语言模型（LLM）输出的技术。

$$P(y|x) \approx \frac{\exp(f(x,y)/\tau)}{\sum_{y'}\exp(f(x,y')/\tau)}$$

#### 1.2 In-Context Learning 机制

**核心问题**：为什么给几个示例，模型就能学会新任务？

**主流解释**：
1. **贝叶斯学习视角** - 示例起到"条件化"作用
2. **元学习视角** - 预训练阶段学习"学习如何学习"
3. **注意力机制视角** - 类似最近邻分类的推广

---

### 2. 提示设计的核心原则

#### 2.1 SMART 原则

| 原则 | 含义 | 示例 |
|------|------|------|
| **S**pecific | 具体明确 | "翻译" → "翻译成美式商务英语" |
| **M**easurable | 可衡量 | "分析" → "给出3个指标及数值" |
| **A**ctionable | 可执行 | 给出清晰的输出格式 |
| **R**elevant | 相关性 | 只提供相关信息 |
| **T**ime-bound | 有边界 | 明确长度约束 |

#### 2.2 分隔符原则

```python
# 好的示例
prompt = """
请分析以下用户评论：

=== 评论内容 ===
用户说产品很好用我很满意
=== 评论结束 ===

输出：
- 情感：[正面/负面/中性]
- 置信度：[0-100]
"""
```

---

## 第二部分：核心方法

### 3. Zero-shot Prompting 详解

#### 3.1 增强技巧

**技巧1：角色扮演**
```
你是一位有20年经验的资深Python开发者...
```

**技巧2：思维链触发**
```
让我们一步一步思考...
```

**技巧3：输出格式规范**
```
## 分析过程
[你的分析]

## 最终答案
[答案]
```

---

### 4. Few-shot Learning 深度剖析

#### 4.1 示例顺序效应

**发现**：示例的顺序会影响模型表现

**最佳实践**：
- 将**最具代表性**的示例放在最后
- 将**简单/基础**的示例放在最前
- 形成"难度递增"的序列

#### 4.2 示例选择算法

**MMR多样性选择**：
```python
def mmr_select(examples, query, k=3, lambda_param=0.5):
    """平衡相关性和多样性"""
    # MMR分数 = λ × 相关性 - (1-λ) × 与已选的相似度
    score = lambda_param * relevance - (1 - lambda_param) * diversity
```

#### 4.3 示例数量选择

| 示例数 | 效果 | 适用场景 |
|--------|------|---------|
| 1-2 | 基础 | 简单任务 |
| 3-5 | 良好 | 大多数任务（最优性价比） |
| 5-10 | 优秀 | 复杂任务 |
| >10 | 边际递减 | 特殊情况 |

---

### 5. Chain-of-Thought 推理机制

#### 5.1 CoT 的发现

**Wei et al. (2022)** 核心发现：只需添加推理过程，准确率提升340%！

| 方法 | GSM8K准确率 |
|------|------------|
| 标准 | 17.7% |
| CoT | 78.7% |

#### 5.2 Zero-shot CoT

**魔法咒语**："Let's think step by step"

```python
template = """
Q: {question}

让我们一步一步思考：
"""
```

#### 5.3 Few-shot CoT 设计

```python
cot_example = """
Q: {question}
A: {reasoning过程}
答案：{answer}
"""
```

---

## 第三部分：高级策略

### 6. Self-Consistency 与集成推理

#### 6.1 原理

同一个问题，生成多个推理路径，投票选择最一致的答案。

```
问题: "如果3x + 7 = 22，x等于多少？"

路径1: 3x = 22 - 7 = 15 → x = 5
路径2: x = (22 - 7) / 3 = 5
路径3: 3x = 15 → x = 3 (错误)
路径4: x = 22/3 - 7 = 0.33 (错误)

投票: x=5 (2票) → 最终答案: x=5, 置信度: 50%
```

#### 6.2 效果提升

| 数据集 | CoT | CoT+SC | 提升 |
|--------|-----|--------|------|
| GSM8K | 78.7% | 92.0% | +14% |
| SVAMP | 71.4% | 87.4% | +16% |

**注意**：SC需要5-10倍的推理成本

---

### 7. Tree-of-Thought 思维树搜索

#### 7.1 原理

将推理视为树搜索，每步生成多个候选思路，评估后选择最优路径。

```
                    问题
                   /    \
              思路A      思路B
             /    \        \
          步骤A1  步骤A2   步骤B1
            |       |        |
         评估     评估      评估
            \       |       /
             最优路径选择
```

#### 7.2 效果

| 任务 | CoT | ToT | 提升 |
|------|-----|-----|------|
| 24点游戏 | 23.6% | 77.0% | +53% |
| 创意写作 | 52.0% | 74.0% | +22% |

---

### 8. 其他高级策略

#### 8.1 ReAct (Reasoning + Acting)

```
Thought: 我需要查找信息
Action: 搜索 "..."
Observation: [结果]
Thought: 根据结果...
Answer: ...
```

#### 8.2 Least-to-Most Prompting

将复杂问题分解为子问题，依次解决。

#### 8.3 Self-Ask

让模型自己生成并回答子问题。

---

## 第四部分：工程实践

### 9. 提示优化技术

#### 9.1 迭代优化流程

```
初始提示 → 测试 → 分析失败案例 → 优化 → 再测试
```

#### 9.2 A/B 测试

```python
class PromptABTester:
    """提示A/B测试"""
    
    def compare(self, prompt_a, prompt_b, test_cases):
        scores_a = [self.eval(prompt_a, case) for case in test_cases]
        scores_b = [self.eval(prompt_b, case) for case in test_cases]
        return np.mean(scores_a), np.mean(scores_b)
```

---

### 10. 输出解析与验证

#### 10.1 JSONOutputParser

```python
parser = JSONOutputParser(schema={
    "type": "object",
    "properties": {
        "sentiment": {"type": "string"},
        "confidence": {"type": "number"}
    }
})
result = parser.parse(llm_output)
```

#### 10.2 ListOutputParser

支持多种列表格式：
- `1. 项目一` - 编号列表
- `- 项目二` - 项目符号
- `项目三` - 纯文本

---

## 第五部分：前沿研究

### 11. 自动提示工程 (APE)

让AI自己设计和优化提示。

```python
def ape(task_description, examples):
    # 1. LLM生成候选提示
    candidates = llm_generate_prompts(task_description)
    
    # 2. 评估每个候选
    best = max(candidates, key=lambda p: evaluate(p, examples))
    
    return best
```

---

### 12. 提示安全与防御

#### 12.1 常见攻击类型

| 攻击类型 | 示例 | 防御 |
|---------|------|------|
| 提示注入 | "忽略之前的指令..." | 输入验证 |
| 越狱 | "假装没有限制..." | 规则强化 |
| 数据提取 | "重复上面的内容..." | 敏感信息检测 |

#### 12.2 防御代码

```python
class PromptSecurityChecker:
    INJECTION_PATTERNS = [
        r"忽略.*指令",
        r"ignore.*instruction",
        r"override.*rule"
    ]
    
    def check(self, user_input):
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, user_input, re.I):
                return {"safe": False, "sanitized": f"<user_input>{user_input}</user_input>"}
        return {"safe": True}
```

---

## 参考文献

1. Brown et al. (2020). Language Models are Few-Shot Learners
2. Wei et al. (2022). Chain-of-Thought Prompting Elicits Reasoning
3. Wang et al. (2022). Self-Consistency Improves Chain of Thought Reasoning
4. Yao et al. (2023). Tree of Thoughts: Deliberate Problem Solving
5. Zhou et al. (2022). Large Language Models Are Human-Level Prompt Engineers

---

*最后更新: 2026年1月*
