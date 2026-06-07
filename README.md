# 基于 AlertBERT 的 LLM 辅助 SOC 告警语义分析

**课程**：网络与信息安全课程实践项目  
**题目**：AI-Based SOC Alert Noise Reduction & Triage (Problem E3A)  
**团队**：Sisyphus  
**日期**：2026 年 6 月

---

## 项目概述

安全运营中心（SOC）每天接收数万条安全告警，但真实攻击仅占 0.01%。本项目以 AlertBERT 自监督聚类为基线，探索 LLM 在告警语义分析中的价值边界。核心发现：**聚类质量决定 LLM 分析上限，Few-shot 范式学习可将攻击阶段识别率从 53.3% 提升至 84.0%，单次分析成本约 $0.00026**。

---

## 目录

1. [一：AIT-ADS-A 数据集与 AlertBERT 基线](#一ait-ads-a-数据集与-alertbert-基线)
2. [二：Completeness 困境](#二completeness-困境)
3. [三：直接 LLM 分析的失败](#三直接-llm-分析的失败)
4. [四：Few-shot 攻击范式学习](#四few-shot-攻击范式学习)
5. [五：结论与关键发现](#五结论与关键发现)
6. [快速开始](#快速开始)
7. [项目结构](#项目结构)
8. [复现实验](#复现实验)

---

## 一：AIT-ADS-A 数据集与 AlertBERT 基线

### AIT-ADS-A 数据集

AIT-ADS（Advanced Intrusion Detection Dataset）包含来自 3 个 IDS 的 2,655,821 条真实 SOC 告警，覆盖 8 个攻击场景，每个场景包含完整的多阶段攻击链。AIT-ADS-A 是 AlertBERT 作者的增强版本，通过噪声注入模拟真实 SOC 中约 0.01% 的攻击率，并将多场景告警打乱为单一数据流。

8 个攻击场景（代号）：

| 场景编号 | 代号 | 攻击类型 | 攻击阶段数 |
|---------|------|---------|-----------|
| 0 | fox | 标准 Web 入侵 | 12 |
| 1 | harrison | 标准 Web 入侵 | 11 |
| 2 | russellmitchell | 标准 Web 入侵 | 11 |
| 3 | santos | 标准 Web 入侵 | 13 |
| 4 | shaw | 标准 Web 入侵 | 12 |
| 5 | wardbeck | 增强 Web 入侵 | 12 |
| 6 | wheeler | 增强 Web 入侵 | 12 |
| 7 | wilson | 标准 Web 入侵 | 11 |

每个场景的攻击阶段包括：服务扫描（service_scan）、Web 扫描（wpscan）、目录枚举（dirb）、WebShell 上传（webshell_cmd）、密码破解（crack_passwords）、DNS 隧道（dnsteal）、权限提升（escalated_sudo_command）、用户切换（attacker_change_user）等。

### AlertBERT 基线表现

AlertBERT 是一个自监督告警分组框架，包含两个阶段：
1. **嵌入阶段**：用掩码语言模型（MLM）在告警字段上训练 BERT，输出语义向量
2. **分组阶段**：基于时间-余弦距离 $d = \max(|\Delta t|, \theta \cdot (1 - \cos(e_1, e_2)))$ 进行密度聚类

在全 8 个 AIT-ADS-A 攻击场景上，AlertBERT 的基线表现如下：

| 指标 | 均值 | 标准差 |
|------|------|--------|
| Purity | 0.9992 | 0.0005 |
| Completeness | 0.4725 | 0.3245 |
| V-Measure | 0.5712 | 0.3205 |
| ARI | 0.4567 | 0.4368 |
| NMI | 0.2624 | 0.1484 |
| 总簇数 | 95,973 | -- |

**关键观察**：Purity 接近完美（99.92%），说明簇内部几乎不会混入不同攻击的告警。但 Completeness 仅 0.47 且波动极大（标准差 0.32），说明同一攻击链的告警被拆散到了多个簇中。

---

## 二：Completeness 困境

### 为什么 Completeness 这么低？

AlertBERT 使用固定的时间窗口参数 $\delta=2.0$ 秒。对于跨数小时的多阶段攻击链，时间距离超过 $\delta$ 的告警被强制拆分到不同簇。这是密度聚类的固有缺陷。

以 scenario 0（fox）为例：

| 指标 | 数值 |
|------|------|
| Baseline 簇数 | 7,702 |
| Completeness | 0.106 |

### 攻击阶段碎片化示例

dirb（目录枚举）阶段包含 4,522 条告警，但由于时间跨度超过 2 秒窗口，被拆散到多个簇中。分析师无法从这些碎片化的簇中看到完整的攻击链。

**根本问题**：AlertBERT 的密度聚类保证了簇内纯净（高 Purity），但无法跨越时间间隙将同一攻击的不同阶段连接起来（低 Completeness）。这是后续 LLM 分析必须面对的现实约束。

---

## 三：直接 LLM 分析的失败

### 实验设计

我们按照原提案设想，在 scenario 0 上进行了全量 LLM 分析实验：
- 模型：DeepSeek-V4-Flash
- Tier 1：对 scenario 0 的全部簇逐一生成中文攻击摘要（风险等级、处置建议、关键 IoC）
- Tier 2：对筛选出的可疑簇进行跨簇攻击链推理

### Tier 1 全量分析结果

| 指标 | 数值 |
|------|------|
| Tier 1 调用次数 | 7,702 |
| Tier 2 调用次数 | 1 |
| 总 LLM 调用次数 | 7,703 |
| 总成本 | $1.2238 |
| 解析失败次数 | 0 |
| Tier 1 单次平均成本 | $0.000159 |
| 总耗时 | 27,409s（约 7.6 小时） |

**风险分布**：

| 风险等级 | 簇数 | 占比 |
|---------|------|------|
| benign | 6,199 | 80.5% |
| low | 512 | 6.6% |
| medium | 867 | 11.3% |
| high | 117 | 1.5% |
| critical | 0 | 0.0% |
| unknown | 7 | 0.1% |

### 关键发现：High-risk 判断几乎全是假阳性

在 117 个被 LLM 判定为 high-risk 的簇中，**116 个（99.1%）的 dominant true label 为 "-"（良性）**。LLM 将正常的 TLS 握手失败、Nmap 扫描、ClamAV 更新等良性告警误判为攻击。

**示例 1：被误判为 high-risk 的良性 TLS 流量**

> Cluster 70（size=56）  
> LLM 输出：风险等级 = high  
> 摘要："短时间内大量 TLS 无效握手与记录告警，疑似扫描或畸形包攻击"  
> 真实标签：全部为 "-"（良性）

该簇实际是 AlertBERT 对正常 TLS 流量的分组，LLM 仅从 "TLS 无效握手" 的表面语义出发给出了 high-risk 判断。

**示例 2：被误判为 high-risk 的邮件服务器正常认证**

> Cluster 238（size=5）  
> LLM 输出：风险等级 = high  
> 摘要："邮件服务器遭受暴力破解攻击，来源 IP 尝试无效用户登录"  
> 真实标签：全部为 "-"（良性）

少量的 Dovecot 认证成功日志被 LLM 误判为暴力破解攻击。

### 为什么直接 LLM 分析所有簇效果有限？

1. **噪声淹没信号**：7,702 个簇中 6,192 个（80.4%）的 dominant true label 是 "-"，真实攻击阶段被大量良性簇稀释
2. **攻击阶段碎片化**：Scenario 0 有约 56,058 条良性告警 vs. 约 6,000 条攻击告警（约 9:1）。webshell 上传仅 3 条、密码破解仅 2 条、权限提升仅 28 条，无法形成足够密度的簇供 LLM 分析
3. **High-risk 假阳性率极高**：99.1% 的 high-risk 判断为假阳性，说明 LLM 容易被告警的表面语义误导

**结论**：直接 LLM 分析所有簇效果有限——攻击信号被噪声淹没，且 LLM 倾向于将表面可疑的良性告警误判为 high-risk。问题的根源在聚类阶段（Completeness 过低导致攻击碎片化），不在 LLM 本身。

---

## 四：Few-shot 攻击范式学习

### LLM 语义分析评判标准

在报告"正确/错误"之前，必须明确我们的评判逻辑：

1. **Ground Truth**：AIT-ADS-A 中每条告警都有标签，非 `"-"` 的标签表示属于某个 attack phase（如 `dirb.0.48`）
2. **Attack-phase 簇**：一个簇的 dominant true label 为非 `"-"`，即该簇主要包含某个攻击阶段的告警
3. **Correct**（识别成功）：LLM 输出的风险等级为 **medium / high / critical**
4. **Missed**（识别失败）：LLM 输出的风险等级为 **benign / low**
5. **注意**：该指标本质上是**攻击识别召回率（Recall）**，而非完整的 Precision/Recall——因为我们只统计了 attack-phase 簇的正确率，没有统计 benign 簇被误判为攻击的 false positive 率

### 关于实验可复现性的说明

由于 LLM 调用设置了 `temperature > 0`，每次运行的具体输出和评判结果可能存在差异。以下表格和案例展示了**代表性结果**，实际复现时攻击识别召回率可能在 ±5% 范围内波动，但 Few-shot 相对 Zero-shot 的显著提升趋势是稳定的。所有量化指标均以 `experiments/smart_split_eval/summary.json` 中的实际运行结果为准。

### 新思路

与其盲目分析所有簇，不如先从历史攻击模式中学习，构建攻击范式模板，再作为 Few-shot 示例指导 LLM 分析新场景。

### 实验设计

| 分组 | 训练场景 | 测试场景 | 攻击链类型 |
|------|---------|---------|-----------|
| Group A | 0, 1, 2, 5 | 3, 4, 6, 7 | 标准/增强 Web 入侵链 |

从训练场景（0, 1, 2, 5）提取 attack-phase 簇样本，经 LLM 总结后生成攻击范式模板（JSON 格式），再作为 Few-shot 示例指导 LLM 分析测试场景。

**攻击范式模板格式示例**（6 个阶段：侦察→初始访问/凭据攻击→执行/持久化→数据外泄→权限提升/横向移动）：

```json
{
  "attack_chain_name": "Web应用典型攻击链",
  "stages": [
    {
      "stage_name": "侦察与信息收集",
      "mitre_tactic": "TA0043 - Reconnaissance",
      "typical_alert_types": ["A-Dns-Chr", "S-Flw-Nmp", "W-Acc-400", ...],
      "alert_text_patterns": [
        "AMiner: New characters in DNS domain.",
        "Suricata: Alert - ET SCAN Possible Nmap User-Agent Observed",
        "Wazuh: Web server 400 error code."
      ],
      "duration_minutes": "10-30",
      "key_indicators": ["大量400错误码来自同一源IP", "新的DNS域名或字符出现"],
      "benign_looking_but_actually_attack": [
        "W-Acc-400 (单次400错误可能为良性，但大量出现则为扫描)"
      ]
    },
    ...
  ],
  "chain_summary": "攻击者首先通过DNS、服务扫描和Web指纹发现目标..."
}
```

范式中每个阶段包含典型告警类型、原始日志文本模式、持续时间、关键指标，以及**最容易被误判为良性的攻击信号**（如 CPU 监控告警实为离线密码破解）。这些范例在 Few-shot prompt 中作为上下文注入，使 LLM 在分析新场景时能参照已知攻击模式进行判断。

评估方法：对每个测试场景的 attack-phase 簇，分别进行 Zero-shot 和 Few-shot 分析。

### 逐场景对比

| 场景 | 分组 | 总簇数 | Attack 阶段数 | Zero-shot | Few-shot | 提升 | 降级 | ZS 成本 | FS 成本 |
|------|------|--------|-------------|-----------|----------|------|------|---------|---------|
| S3 (santos) | A | 18,188 | 13 | 55.0% (11/20) | 90.0% (18/20) | +8 | -1 | $0.0078 | $0.0127 |
| S4 (shaw) | A | 24,364 | 12 | 61.1% (11/18) | 88.9% (16/18) | +5 | 0 | $0.0090 | $0.0137 |
| S6 (wheeler) | A | 6,681 | 12 | 45.0% (9/20) | 80.0% (16/20) | +7 | 0 | $0.0070 | $0.0126 |
| S7 (wilson) | A | 3,724 | 11 | 52.9% (9/17) | 76.5% (13/17) | +6 | -2 | $0.0067 | $0.0115 |

### 汇总结果

| 指标 | Zero-shot | Few-shot | 净提升 |
|------|-----------|----------|--------|
| 总成本 | -- | $0.0808 | -- |
| 攻击识别召回率 | 53.3% (40/75) | 84.0% (63/75) | +23 个簇 |
| 升级数 | -- | 26 | -- |
| 降级数 | -- | 3 | -- |

### 全簇精度验证（Scenario 7）

上述汇总结果仅统计了 attack-phase 簇的识别率（Recall）。为了更严谨地验证 Few-shot 在真实 SOC 场景中的效果，我们在 Scenario 7 上对 **全部 3,724 个簇** 进行了完整的 Precision / Recall / F1 / Accuracy 评估。

**指标说明**：
- **Precision（精确率）**：被 LLM 判定为 "攻击" 的簇中，实际真的是攻击的比例。反映 LLM 判断的可信度
- **Recall（召回率）**：实际为攻击的簇中，被 LLM 正确识别为 "攻击" 的比例。反映攻击漏报情况
- **F1 Score**：Precision 和 Recall 的调和平均，综合衡量识别质量
- **Accuracy（准确率）**：所有簇中被正确分类（攻击判攻击、良性判良性）的比例
- **False Positive（FP）**：实际是良性但被误判为攻击的簇数，反映噪声误报程度

**Scenario 7 全簇评估结果**：

| 指标 | Zero-shot | Few-shot | 变化 |
|------|-----------|----------|------|
| Precision | 1.12% | 3.36% | **+2.24%** |
| Recall | 46.67% | 60.00% | **+13.33%** |
| F1 Score | 2.19% | 6.36% | **+4.17%** |
| Accuracy | 83.24% | 92.88% | **+9.64%** |
| False Positive | 616 | 259 | **-357 (-58%)** |
| Cost | $0.615 | $0.973 | 总计 $1.588 |

**关键发现**：
1. **Precision 极低**：即使 Few-shot 也只有 3.36%，说明直接分析所有簇会产生海量假阳性。每 100 个被判定为攻击的簇中，只有约 3 个是真正的攻击
2. **Recall 提升明显**：Few-shot 帮助识别了更多真实攻击（60% vs 46.67%），验证了范式学习的有效性
3. **FP 大幅减少**：Few-shot 将误报从 616 降到 259，减少 **58%**，说明范式中的 "看起来良性但实际是攻击" 的提示起到了关键作用
4. **Accuracy 大幅提升**：从 83% → 93%，因为大量 benign 簇被正确识别

**结论**：全簇评估精确量化了核心论点——在 0.01% 攻击率的 SOC 场景中，直接 LLM 分析所有簇的 Precision < 4%，不具备工程可用性。Few-shot 虽能显著降低假阳性（-58%）并提升 Recall（+13.33%），但 precision 仍然过低。这进一步验证了：**必须先改善上游聚类质量，再谈 LLM 语义分析**。

### LLM 输出对比：同一簇的 Zero-shot vs Few-shot

**案例 1：crack_passwords.0.50（Scenario 4，8 条告警）**

Zero-shot 输出：
> 风险等级：**low**  
> 摘要："短时间内DNS查询到.biz TLD，伴随Dovecot认证成功，疑似侦察或异常流量"

Few-shot 输出：
> 风险等级：**high**  
> 摘要："疑似通过DNS侦察和邮件认证进行初始访问，伴随CPU异常升高，可能为凭据攻击或数据外泄前兆"

分析：Zero-shot 仅识别为可疑侦察（low），Few-shot 通过范式中的 "CPU 异常 → 密码破解" 关联模板，将风险等级提升至 high，正确识别为凭据攻击阶段。

**案例 2：attacker_change_user.0.74（Scenario 6，10 条告警）**

Zero-shot 输出：
> 风险等级：medium  
> 摘要："疑似用户提权操作（UID变更）伴随审计日志新模式"

Few-shot 输出：
> 风险等级：**high**  
> 摘要："疑似权限提升攻击：短时间内出现UID变更及多个新审计参数组合，符合横向移动/提权模式"

分析：Few-shot 将风险等级从 medium 提升至 high，并增加了攻击链阶段定位（"横向移动/提权"），更准确地反映了攻击严重性。

**案例 3：dnsteal.2.67（Scenario 6，2 条告警）**

Zero-shot 输出：
> 风险等级：**benign**  
> 摘要："审计日志中检测到服务启停参数变化，可能为正常运维操作"

Few-shot 输出：
> 风险等级：**medium**  
> 摘要："内部主机internal_share上出现新的服务启停参数组合，可能为后门安装或安全服务关闭"

分析：Zero-shot 将新的审计参数组合误判为正常运维，Few-shot 通过范式中 "审计日志新参数 → 后门安装或安全服务关闭" 的关联，正确识别为数据外泄阶段的前置行为。

### 仍然存在的失败案例

**失败案例 1：crack_passwords.0.73（Scenario 6，1 条告警）**

> Zero-shot："监控日志中CPU值偏离平均值，属于性能异常或管理性事件，无攻击迹象。"（risk=benign）  
> Few-shot："可疑CPU异常波动，可能为离线密码破解或恶意进程，但缺乏关联证据"（risk=low）

Few-shot 虽修正了攻击意图（从"正常"到"可疑破解"），但风险等级仅提升至 low（真实应为 medium/high）。单条孤立 CPU 告警缺少上下文关联，即使是 Few-shot 范例也无法提供足够证据支撑更高风险等级。

**失败案例 2：escalated_sudo_command.0.75（Scenario 6，3 条告警）**

> Zero-shot："正常的管理员sudo提权操作，无攻击迹象"（risk=benign）  
> Few-shot："疑似通过登录会话获取凭据后sudo提权至root"（risk=medium）

Few-shot 成功识别了提权行为的可疑性（从 benign 提升至 medium），但未能达到 high 等级。根因：`sudo` 提权在审计日志中表现为正常管理行为的语义特征极强，LLM 难以区分合法运维与恶意提权。

**失败案例 3：wpscan.0.47（Scenario 4，9,663 条告警）**

> Zero-shot："针对内网Web服务器的大规模HTTP错误请求攻击或扫描"（risk=high）  
> Few-shot："内网IP 192.168.131.215 对 Web 服务器发起 HTTP 侦察扫描，产生大量 400 错误码"（risk=medium）

这是一个**降级案例**：Zero-shot 正确识别为 high-risk 的大规模扫描攻击，但 Few-shot 受范式中 "扫描探测常归类为侦察/低风险" 的模板影响，将风险等级降级为 medium。说明 Few-shot 范例在提升大部分攻击识别率的同时，也可能对特定大型扫描簇产生过度保守的判断。

### 成本分析

| 指标 | 数值 |
|------|------|
| 总 LLM 调用次数 | 310（4 个场景，ZS+FS 各半） |
| 总成本 | $0.0810 |
| 单次分析成本 | ~$0.00026 |

在 SOC 场景下，对 73 个 attack-phase 簇进行精确分诊的成本低于 4 美分，具备工程可行性。

---

## 五：结论与关键发现

### 核心发现

1. **AlertBERT 高纯度但低完整性**：Purity=0.9992 说明簇内部纯净，但 Completeness=0.4725 说明攻击阶段被时间窗口碎片化
2. **直接 LLM 分析所有簇效果差**：117 个 high-risk 判断中 116 个（99.1%）为假阳性，攻击信号被噪声淹没
3. **Few-shot 范式学习显著提升攻击识别召回率**：从 53.3% 提升至 84.0%，净增 23 个正确识别的攻击阶段簇
4. **成本极低**：$0.0808 覆盖 4 个完整场景，单次分析约 $0.00026
5. **剩余挑战**：语义相似的良性/攻击模式（sudo 提权、CPU 监控）即使 Few-shot 也容易误判

### 未来工作方向

1. **自适应时间窗口**：根据告警密度动态调整 $\delta$，而非全局固定
2. **对比学习微调**：在 AlertBERT 嵌入上增加对比学习损失，使同一攻击阶段的告警在嵌入空间中更近
3. **聚类前过滤**：通过规则/签名过滤已知良性告警，减少 10:1 的噪声比
4. **动态范例库**：为每类攻击链维护 10-20 个典型簇摘要，定期用新场景校准风险阈值

### 核心启示

> **先解决"聚类能否把攻击分出来"，再谈"LLM 能否把攻击讲清楚"。**  
> 盲目引入 LLM 而不改善上游聚类质量，只会生成漂亮的假阳性报告。

---

## 快速开始

### 环境要求

- Python 3.10+
- NVIDIA GPU（推荐 4090 48GB，使用 `CUDA_VISIBLE_DEVICES=0`）
- DeepSeek API Key（用于 LLM 分析实验）

### 数据集准备

AIT-ADS-A 数据集体积较大（约 600MB），**不包含在代码仓库中**，需按以下步骤放置：

**方式一：从 Zenodo 下载原始 AIT-ADS，再用 AlertBERT 脚本生成增强版本**

1. 下载 AIT-ADS 原始数据：
   ```bash
   # 前往 Zenodo 下载: https://zenodo.org/record/XXXXXXX
   # 解压后得到 alerts_json/ 和 alerts_csv/ 目录
   ```

2. 使用 AlertBERT 脚本生成 AIT-ADS-A：
   ```bash
   cd AlertBERT
   python build_augment_files.py --input-dir alerts_json --output-dir aitads_augmented
   cd ..
   ```

**方式二：直接拷贝已生成的 AIT-ADS-A（如果你有预先生成的版本）**

```bash
# 将 aitads_augmented/ 目录放置在 AlertBERT/ 下
cp -r /path/to/your/aitads_augmented AlertBERT/
```

**验证数据就位**：
```bash
ls AlertBERT/aitads_augmented/scenario_*.json  # 应看到 8 个场景的 JSON 文件
```

### 安装步骤

```bash
# 1. 进入项目目录
cd submission

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行设置脚本
bash setup.sh

# 4. 应用 AlertBERT 模型补丁
cd AlertBERT && git apply ../patches/alertbert_models.patch && cd ..
```

### 配置 LLM

```bash
export DEEPSEEK_API_KEY="your-api-key-here"
```

---

## 项目结构

```
submission/
├── README.md                    # 本文件
├── requirements.txt             # Python 依赖
├── setup.sh                     # 环境设置脚本
├── patches/
│   └── alertbert_models.patch   # AlertBERT 模型补丁
├── hybrid_pipeline/             # 混合流水线核心代码
│   ├── config.py                # 配置管理
│   ├── pipeline.py              # 主流水线
│   ├── evaluate.py              # 评估指标
│   ├── cluster_sampler.py       # 簇特征提取与采样
│   ├── llm_refine.py            # LLM 优化器
│   ├── llm_attack_analyzer.py   # LLM 攻击分析器
│   ├── alert_type_parser.py     # 告警类型解析器
│   └── __init__.py
├── AlertBERT/                   # AlertBERT 基线代码
│   ├── alertbert/
│   │   └── models.py            # 应用了补丁的模型
│   ├── server_configs/          # 场景配置文件（8 个场景）
│   └── requirements.txt
├── scripts/                     # 实验脚本
│   ├── run_baseline.py          # 运行 AlertBERT 基线
│   ├── run_llm_analysis.py      # 运行 LLM 全量分析
│   ├── evaluate_all_scenarios.py # 全场景评估
│   ├── evaluate_smart_split.py  # Smart Split 评估（ZS vs FS）
│   ├── evaluate_precision.py    # 全簇精度评估（Precision/Recall/F1）
│   └── build_attack_paradigm.py # 构建攻击范式模板
└── experiments/                 # 实验结果
    ├── baseline_original/       # 基线结果
    │   └── summary.json         # Purity=0.9992, Completeness=0.4725
    ├── smart_split_eval/        # Few-shot 跨场景评估
    │   ├── summary.json         # 汇总：ZS=53.3%, FS=84.0%, cost=$0.0808
    │   ├── scenario_3/results.json
    │   ├── scenario_4/results.json
    │   ├── scenario_6/results.json
    │   └── scenario_7/results.json
    ├── all_scenarios_eval/      # 全场景评估
    ├── llm_analysis_s0/         # Scenario 0 LLM 全量分析
    │   └── metadata.json        # 7703 calls, $1.2238, ~7.6h
    └── attack_paradigm/         # 攻击范式模板
```

---

## 复现实验

### 1. 运行 AlertBERT 基线

```bash
python scripts/run_baseline.py
```

输出结果保存在 `experiments/baseline_original/summary.json`。

预期结果：
- Purity: 0.9992 +/- 0.0005
- Completeness: 0.4725 +/- 0.3245
- V-Measure: 0.5712 +/- 0.3205
- ARI: 0.4567 +/- 0.4368
- NMI: 0.2624 +/- 0.1484
- 总簇数: 95,973

### 2. 运行 LLM 全量分析（Scenario 0）

```bash
# 分析全部簇（7702 个，成本约 $1.22，耗时约 7-8 小时）
python scripts/run_llm_analysis.py --scenario 0

# 或限制分析数量以控制成本（693 个簇，成本约 $0.15，耗时约 50 分钟）
python scripts/run_llm_analysis.py --scenario 0 --max-clusters-tier1 693
```

对 scenario 0 的簇进行 Tier 1 逐簇分析 + Tier 2 跨簇推理。

输出结果保存在 `experiments/llm_analysis_s0/metadata.json`。

预期结果（全量分析）：
- Tier 1 调用: 7,702 次
- Tier 2 调用: 1 次
- 总成本: $1.2238
- 解析失败: 0 次
- 总耗时: ~27,409s（约 7.6 小时）
- 风险分布: benign 80.5%, low 6.6%, medium 11.3%, high 1.5%
- High-risk 假阳性率: ~99.1%（116/117 的 high-risk 簇 dominant true label 为 "-"）

### 3. 构建攻击范式

```bash
python scripts/build_attack_paradigm.py --scenarios 0 1 2 5
```

从训练场景提取攻击阶段样本，生成 Few-shot 范例。输出保存在 `experiments/attack_paradigm/`。

### 4. 运行 Smart Split 评估

```bash
python scripts/evaluate_smart_split.py --test-scenarios 3 4 6 7
```

对测试场景分别进行 Zero-shot 和 Few-shot 分析，输出对比结果。

预期结果（汇总）：
- 总成本: $0.0808
- Zero-shot 攻击识别召回率: 53.3%
- Few-shot 攻击识别召回率: 84.0%
- 净提升: +23 个簇
- 升级: 26, 降级: 3

各场景详细结果保存在 `experiments/smart_split_eval/scenario_N/results.json`。

### 5. 运行全簇精度评估（Precision/Recall/F1）

```bash
python scripts/evaluate_precision.py \
    --scenario 7 \
    --paradigm experiments/attack_paradigm/attack_paradigm.json \
    --output-dir experiments/precision_s7 \
    --max-cost 2.0
```

对单个场景的全部簇进行 Zero-shot 与 Few-shot 分析，计算完整的 Precision、Recall、F1、Accuracy 指标。

**注意**：此实验需要分析所有簇（非仅 attack-phase 簇），成本较高。

预期结果（Scenario 7）：
- 总簇数: 3,724
- Zero-shot 成本: ~$0.615, Few-shot 成本: ~$0.973
- Precision: 1.12% (ZS) → 3.36% (FS)
- Recall: 46.67% (ZS) → 60.00% (FS)
- F1: 2.19% (ZS) → 6.36% (FS)
- Accuracy: 83.24% (ZS) → 92.88% (FS)
- False Positive: 616 (ZS) → 259 (FS)

完整报告保存在 `experiments/precision_s7/precision_report.json`。

---

## 参考文献

[1] Yang, L., et al. "True attacks, attack attempts, or benign triggers? an empirical measurement of network alerts in a security operations center." USENIX Security, 2024.

[2] Turcotte, M. A. H., et al. "Automated Alert Classification and Triage (AACT)." arXiv:2505.09843, 2025.

[3] Karner, L., et al. "AlertBERT: A noise-robust alert grouping framework for simultaneous cyber attacks." arXiv:2602.06534, 2026.

[4] Ede, T. v., et al. "DEEPCASE: Semi-Supervised Contextual Analysis of Security Events." IEEE S&P, 2022.

[5] Wei, B., et al. "CORTEX: Collaborative LLM Agents for High-Stakes Alert Triage." arXiv:2510.00311, 2025.
