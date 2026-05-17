# 模板文档检测

## 目标

给定一份电子文档，判断其是否由模板生成。如果是由模板生成的，进一步识别其所属的模板类型（template_id）。

## 核心问题

模板检测的本质是区分 **人工编写文档** 与 **模板生成文档**，而非对文档做精细语义理解。关键洞察：

- 模板生成文档具有高度结构化特征：固定文本区域多、可变字段占比小且位置固定、布局重复性高
- 人工编写文档则文本自由、格式灵活、重复模式少

## 检测方案

### 方案 A：基于相似度的聚类检测（推荐）

1. **特征提取**
   - 布局特征：文本块位置、大小、对齐方式
   - 文本特征：固定文本占比、特殊字符比例（数字/标点）、字段模式重复度
   - 结构特征：段落数、列表/表格结构、层级深度
2. **模板库构建**
   - 对已知模板生成文档提取特征向量，建立模板库
   - 每个模板类型对应一个或多个特征质心
3. **检测流程**
   - 提取待检测文档特征向量
   - 计算与模板库中各质心的相似度（余弦相似度 / 欧氏距离）
   - 若最高相似度 > 阈值 → 判定为模板生成，返回对应 template_id
   - 否则 → 判定为人工编写

### 方案 B：基于规则的模式匹配

定义模板文档的常见特征规则：

- 固定文本占比超过阈值（如 >60% 的文本在多个文档中完全相同）
- 包含大量结构化字段占位符（`{{name}}`, `${amount}` 等模式）
- 文档布局高度规律（相同的页眉页脚、表格结构等）

规则可组合打分，超过总分阈值即判定为模板。

### 方案 C：基于异常检测

将人工编写文档视为"正常"，模板生成文档视为"异常"，使用 One-Class SVM / Isolation Forest 等无监督方法建模。适用于只有人工文档样本的场景。

## 接口定义

```python
class TemplateDetector:
    def detect(self, document: Document) -> DetectionResult:
        """
        判断文档是否由模板生成。
        
        Args:
            document: 统一封装的文档对象（包含文本、布局、元信息）
        
        Returns:
            DetectionResult:
                - is_template: bool
                - template_id: str | None
                - confidence: float  # 0~1
                - matched_features: dict  # 匹配到的特征详情
        """
        ...
```

## 评价指标

| 指标 | 说明 |
|------|------|
| 精确率 (Precision) | 判定为模板的文档中，真正是模板的比例 |
| 召回率 (Recall) | 真正的模板文档中被正确判出的比例 |
| F1-score | 精确率与召回率的调和平均 |
| 误报率 (FPR) | 人工文档被误判为模板的比例 |

## 设计原则

- **无监督优先**：尽量减少对标注数据的依赖，优先使用无监督或弱监督方法
- **可扩展**：新模板类型无需重训练即可增量添加
- **可解释**：检测结果应能回溯到具体的匹配特征，便于调试和调优
- **渐进增强**：先从简单的规则/关键词匹配入手，逐步引入更复杂的相似度/聚类方法

## 当前阶段任务

1. 定义 `Document` 统一数据结构（文本、布局、元信息）
2. 实现基础版 `TemplateDetector`：
   - 基于关键词/正则规则匹配的初步检测
   - 基于 TF-IDF + 余弦相似度的模板匹配
3. 设计模板库（`TemplateLibrary`）的存储与加载

## 数据结构草案

```python
@dataclass
class Document:
    raw_text: str              # 纯文本内容
    segments: list[TextSegment]  # 文本块列表（含位置信息）
    metadata: dict             # 来源、格式、页数等

@dataclass
class TextSegment:
    text: str
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    font_size: float | None
    is_heading: bool = False

@dataclass
class DetectionResult:
    is_template: bool
    template_id: str | None
    confidence: float
    matched_features: dict
```

## 依赖关系

```
TemplateDetector
  ├── FeatureExtractor       # 文本/布局特征提取
  ├── TemplateLibrary        # 模板库（存储/加载已知模板特征）
  │     └── TemplateProfile  # 单个模板的特征描述
  └── SimilarityScorer       # 相似度计算引擎
```
