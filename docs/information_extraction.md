# 预定义规则的信息提取

## 目标

对于已识别为模板生成的文档，根据预定义的提取规则，从中抽取出结构化信息，并进行校验。

## 核心概念

对于每个已识别的模板类型，预定义一套提取规则，包含三个核心组件：

### 字段定位器 (Locator)
定位目标字段在文档中的位置：
- **regex**: 正则表达式匹配文本
- **keyword**: 关键词附近查找
- **xpath/css**: HTML/XML 文档的路径定位
- **coordinate**: 基于坐标区域定位（主要针对 PDF）
- **anchor**: 以某个锚点文本为参照的相对定位

### 字段提取器 (Extractor)
从定位到的片段中提取值，支持类型转换：
- **string**: 直接取文本
- **decimal**: 数字提取（处理千分位、货币符号）
- **date**: 日期解析（支持多种格式）
- **enum**: 枚举值映射
- **list**: 列表提取（表格行、重复项）
- **composite**: 组合多个子字段

### 字段校验器 (Validator)
对提取值进行校验：
- **required**: 非空校验
- **range**: 数值范围
- **date_range**: 日期范围
- **enum**: 枚举值白名单
- **regex**: 格式校验
- **custom**: 自定义校验函数

## 规则配置格式

规则以声明式配置定义（YAML）：

```yaml
template_id: "bank_statement_2024"
name: "某银行信用卡账单(2024版)"
detection:
  match_keywords: ["信用卡账单", "还款日"]
  min_similarity: 0.85
fields:
  - name: "statement_date"
    locator:
      type: regex
      pattern: "账单日期[：:]?(\\d{4}-\\d{2}-\\d{2})"
    extractor:
      type: date
      format: "YYYY-MM-DD"
    validator:
      required: true
      type: date_range
      min: "2024-01-01"
      max: "2024-12-31"
  - name: "total_amount"
    locator:
      type: regex
      pattern: "本期应还款金额[：:]?[¥￥]?(\\d+\\.?\\d*)"
    extractor:
      type: decimal
    validator:
      required: true
      min: 0
```

## 校验与后处理

- **字段级校验**：每个字段独立校验（格式、范围、必填等）
- **记录级校验**：跨字段逻辑校验（如金额合计 = 各明细之和）
- **异常处理**：校验失败时记录原因，可选择降级处理或标记人工审核

## 结构化输出

最终输出统一格式的结构化数据：

```yaml
output:
  format: json
  schema:
    template_id: string
    extracted_at: datetime
    source: string    # 原始文档标识
    fields: object    # 提取的字段键值对
    confidence: float # 整体提取置信度
    errors: string[]  # 校验失败信息
```

## 接口定义

```python
class ExtractionRule:
    template_id: str
    fields: list[FieldRule]

class FieldRule:
    name: str
    locator: Locator
    extractor: Extractor
    validator: Validator | None

class RuleEngine:
    def extract(self, document: Document, rules: ExtractionRule) -> ExtractionResult:
        ...
```

## 依赖关系

```
RuleEngine
  ├── FieldLocator      # 字段定位器（regex / keyword / xpath / coordinate）
  ├── FieldExtractor    # 字段提取器（string / decimal / date / enum / list）
  ├── FieldValidator    # 字段校验器（required / range / enum / regex）
  └── RuleLoader        # 规则配置加载器（YAML / JSON）
```
