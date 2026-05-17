# 模板文档检测

## 目标

给定一份电子文档，判断其是否由模板生成。如果是由模板生成的，进一步识别其所属的模板类型（template_id）。

## 核心问题

模板检测的本质是区分 **人工编写文档** 与 **模板生成文档**，而非对文档做精细语义理解。关键洞察：

- 模板生成文档具有高度结构化特征：固定文本区域多、可变字段占比小且位置固定、布局重复性高
- 人工编写文档则文本自由、格式灵活、重复模式少

## 方案限制

- **无预训练/无预标注数据**：系统上线后直接面对未知文档，不存在离线预训练阶段，只能在处理过程中积累和学习
- **仅二元反馈**：对于分类结果，用户只能确认"对/错"，无法提供正确的 template_id。这意味着如果判定为模板 A 但用户说"错"，无法区分是"不是模板"还是"是模板但类型不对"
- **数据集仅用于模拟反馈**：项目提供的数据集可模拟上线后用户的确认结果，用于评估方案有效性，但不可用于预训练
- **不使用 LLM**：不能依赖外网模型，也没有本地 GPU 集群，只能使用传统 ML/DL 或可在普通 CPU 服务器上运行的算法
- **异步学习**：数据积累和模板库更新可以分离 —— 在线即时积累数据，空闲时批量学习更新

## 方案设计

### 整体架构：两阶段管线

```
输入文档
   │
   ▼
┌─────────────────────┐
│  Stage 1            │
│  模板性检测          │  ← 规则驱动，零训练数据即可运行
│  (Template-ness)     │
└─────────┬───────────┘
          │
     ┌────┴────┐
     ▼         ▼
  非模板       是模板
     │         │
     │         ▼
     │    ┌─────────────────────┐
     │    │  Stage 2            │
     │    │  模板识别            │  ← 增量聚类，在线匹配 + 离线学习
     │    │  (Template ID)       │
     │    └─────────┬───────────┘
     │              │
     │         ┌────┴────┐
     │         ▼         ▼
     │     已识别      未匹配
     │       │         │
     │       │         ▼
     │       │     新建模板候选项
     │       │
     ▼       ▼
  输出结果
     │
     ▼
  用户二元反馈 (对/错)
     │
     ▼
  积累到待学习队列
     │
     ▼  (空闲时批量处理)
  ┌──────────────────────┐
  │  离线学习引擎          │
  │  ● Stage1 规则调优     │
  │  ● Stage2 增量聚类     │
  │  ● 模板库更新          │
  └──────────────────────┘
```

### Stage 1：模板性检测

目标：判断文档是否由模板生成，输出 `is_template: bool` + `confidence: float`。

基于规则的启发式打分，每个维度输出 0~1 分，加权求和得到最终模板性分数：

| 特征维度 | 计算方式 | 说明 |
|----------|----------|------|
| **固定文本占比** | 去重 n-gram 占比 / 总文本长度 | 模板文档中大量文本跨文档重复 |
| **布局规律性** | 文本块对齐一致性、行间距方差倒数 | 模板文档布局高度规整 |
| **特殊字符密度** | 数字、标点、分隔符占总字符比例 | 模板文档含大量结构化字段（金额、日期、编号） |
| **重复模式密度** | 正则匹配到的类字段模式（`\d{4}-\d{2}-\d{2}` 等） | 模板文档中日期、金额等模式反复出现 |
| **文本熵** | 字符级/词级信息熵 | 模板文档中固定部分导致熵偏低 |

**判定规则**：

```
template_ness = Σ(w_i * score_i)  // 加权求和
if template_ness > threshold_high  → 判定为模板
if template_ness < threshold_low   → 判定为非模板
if threshold_low ≤ template_ness ≤ threshold_high → 低置信度，标记待确认
```

**设计要点**：
- 零数据即可运行，所有特征无需训练
- 权重和阈值初期可手工设定，后续通过离线学习优化
- 产出 confidence 供下游决策（如自动通过 / 标记人工审核）

### Stage 2：模板识别

目标：对 Stage 1 判为模板的文档，识别其具体模板类型。

#### 在线匹配

1. 提取文档"指纹"特征向量（文本 TF-IDF + 布局特征）
2. 与模板库中已有 `TemplateProfile` 计算相似度
3. 最高相似度 > 阈值 → 返回对应 `template_id`
4. 否则 → 创建一个新的模板候选项（临时 ID），积累足够同类文档后经离线聚类确认

#### 离线学习

利用积累的文档 + 用户反馈，定期（如每日/每周）批量执行：

**步骤 1：确认模板文档池**
- 从待学习队列中筛选出用户确认为"对"的模板文档
- 对于用户确认为"错"的文档：若其 `is_template=true`，则降级为待定或丢弃（因无法确定正确模板类型）

**步骤 2：聚类发现模板类型**
- 对确认的模板文档集合做聚类（如 DBSCAN / HDBSCAN，无需预设 K 值）
- 每个簇对应一个模板类型
- 对于规模达到阈值的簇，创建/更新 `TemplateProfile`
- 噪点文档保留为候选项，待后续更多数据

**步骤 3：规则调优**
- 利用累计数据优化 Stage 1 的权重和阈值
- 方法：在积累数据上搜索最优参数组合（网格搜索 / 贝叶斯优化）

#### 自我演化示例

```
T0: 系统上线，模板库为空
    → 所有文档进入 Stage 1，依赖规则打分判定模板性
    → 判为模板的文档在 Stage 2 均进入"新建候选项"

T1: 积累了一批经用户确认为"对"的模板文档
    → 离线聚类发现 3 个簇 → 创建 3 个 TemplateProfile
    → 更新模板库

T2: 新文档进入 Stage 2，与 3 个已知模板匹配
    → 命中 → 返回模板 ID
    → 未命中 → 继续创建候选项

T3: 积累更多数据，再次离线聚类
    → 原有 3 个模板得到强化，新增 1 个模板
    → Stage 1 阈值根据反馈数据自动微调
```

### 反馈处理策略

| 系统输出 | 用户反馈 | 处理逻辑 |
|----------|----------|----------|
| `{is_template: true, id: "A"}` | 对 | 确认文档归入模板 A，加入待学习队列 |
| `{is_template: true, id: "A"}` | 错 | 无法确定是"非模板"还是"模板类型错误"，标记为待定，不用于模板学习 |
| `{is_template: false}` | 对 | 确认非模板，丢弃 |
| `{is_template: false}` | 错 | 确认是模板但未识别出类型，加入待学习队列（作为模板候选项，但 template_id 未知） |

## 数据结构

```python
@dataclass
class Document:
    raw_text: str
    segments: list[TextSegment]
    metadata: dict

@dataclass
class TextSegment:
    text: str
    bbox: tuple[float, float, float, float]
    font_size: float | None
    is_heading: bool = False

@dataclass
class DetectionResult:
    is_template: bool
    template_id: str | None
    confidence: float
    score_breakdown: dict[str, float]  # 各维度得分明细

@dataclass
class TemplateProfile:
    template_id: str
    feature_vector: np.ndarray        # 聚类质心
    text_fingerprint: dict            # TF-IDF 特征
    layout_signature: dict            # 布局特征
    sample_count: int                 # 确认文档数
    first_seen: datetime
    last_updated: datetime

@dataclass
class Feedback:
    document_id: str
    system_output: DetectionResult
    user_correct: bool                # True=对, False=错
    timestamp: datetime
```

## 接口定义

```python
class TemplateDetector:
    def detect(self, document: Document) -> DetectionResult:
        """两阶段检测：先判模板性，再识别模板类型。"""

    def record_feedback(self, feedback: Feedback) -> None:
        """记录用户反馈，加入待学习队列。"""

    def learn(self) -> LearnReport:
        """离线学习：聚类发现模板 + 规则调优。返回本次学习报告。"""

class TemplateLibrary:
    def match(self, document: Document) -> tuple[str | None, float]:
        """与已知模板匹配，返回 (template_id, similarity)。"""

    def add_candidate(self, document: Document) -> str:
        """创建新的模板候选项，返回临时 ID。"""

    def cluster_and_update(self, documents: list[Document]) -> list[TemplateProfile]:
        """对文档集合做聚类，发现/更新模板类型。"""

class TemplateNessScorer:
    def score(self, document: Document) -> tuple[float, dict[str, float]]:
        """计算模板性分数，返回 (总分, 各维度明细)。"""
```

## 评价指标

| 指标 | 说明 |
|------|------|
| 模板性检测精确率 | 判定为模板的文档中，真正是模板的比例 |
| 模板性检测召回率 | 真正的模板文档中被正确判出的比例 |
| 模板识别准确率 | 已判为模板的文档中，template_id 识别正确的比例 |
| F1-score | 综合衡量 |
| 误报率 (FPR) | 人工文档被误判为模板的比例 |
| 模板库覆盖率 | 已知模板类型占实际模板类型总数的比例（随学习时间增长） |

## 设计原则

- **冷启动**：系统在零训练数据下即可运行
- **增量学习**：模板库随时间自动丰富，无需重新全量训练
- **异步更新**：在线检测路径与离线学习路径解耦
- **弱监督兼容**：仅利用二元反馈信号，不强求完整标注
- **可解释**：检测结果可回溯到各维度得分明细
- **CPU 友好**：所有算法可在普通 CPU 服务器上运行

## 依赖关系

```
TemplateDetector
  ├── TemplateNessScorer     # Stage 1：模板性打分
  │     └── FeatureComputer  # 各维度特征计算
  ├── TemplateLibrary        # 模板库管理
  │     ├── TemplateMatcher  # 在线匹配
  │     └── TemplateLearner  # 离线聚类学习
  └── FeedbackStore          # 用户反馈存储与待学习队列
```

## 当前阶段任务

1. 定义核心数据结构：`Document`、`TextSegment`、`DetectionResult`、`Feedback`
2. 实现 `TemplateNessScorer`：完成 5 个维度的特征计算与加权打分
3. 实现 `TemplateLibrary` 的基础存储与加载
4. 实现 `TemplateMatcher`：基于 TF-IDF + 余弦相似度的在线匹配
5. 实现 `TemplateDetector.detect()` 两阶段流程串联
