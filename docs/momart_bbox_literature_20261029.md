# MoMaRT：VLA / Diffusion Policy 中的检测框与对象表示调研

飞书报告：https://feishu.cn/docx/NxF3dQohVoUqiGxEMAJc57j4nCh

检索日期：2026-10-29。范围：框作为视觉提示、ROI/坐标作为对象token、多对象记忆、在线子目标选择。依据作者论文、项目页及官方仓库；下文区分论文事实与针对MoMaRT的设计建议。未运行第三方模型或新增训练。

结论：已有直接相关方法，但“使用检测框”包含不同接口。框可以表示用户指定的目标，也可以表示一组候选对象，还可以由上层规划器随阶段更新。它们对在线检测、历史信息和目标选择的要求不同，不能仅按是否加框判断效果。

| 工作 | 类型与框的用途 | 多物体/在线目标由谁处理 | 对MoMaRT最相关的边界 |
|---|---|---|---|
| BBox-DP，2026预印本 | DP；原图、画框图、带框首帧作为条件 | 检测模块给出目标；首帧维持目标指代 | 有漏框/错框增强；首帧锚定不等于自动换阶段 |
| Point-VLA，2025预印本 | VLA；带框首帧是额外视觉指令 | 人画框，或模型解释人的指向 | 解决“指哪个”，不能当自主阶段选择证据 |
| VP-VLA，2026预印本 | VLA；上层生成十字点/目标区域框 | 夹爪事件触发VLM更新子任务 | 接近阶段切换，但夹爪变化不能保证任务完成 |
| Scene Graphs，CVPR 2026 | DP；多个对象的外观、2D框、3D位置 | 任务相关对象集合、跟踪与结构化记忆 | 适合部分可见长任务；引入历史与几何依赖 |
| VIMA，ICML 2023 | 多模态提示策略；ROI和框编码成对象token | 多对象集合与交互历史共同进入策略 | 检测器经过领域微调；不是原版DP |
| VIOLA，2022预印本 | Transformer行为克隆；ROIAlign与位置编码 | 多候选框、全景、腕部图像及历史 | 与当前ROI+位置最接近，但动作头是GMM |
| Oat-VLA，2025预印本 | VLA；对象mask聚合与夹爪局部patch | 多对象表示，额外保留接触区域 | 对象分支不是通用检测框裁图；不能混称 |
| ObjectDiff，2026期刊论文 | 扩散策略；检测空间注意力 | 指令与视觉条件；随机撤去检测提示 | 可借鉴检测条件dropout；全文实现未核验 |

下面各项给出对应一手来源与证据位置。表中的MoMaRT边界是根据这些设计作出的判断，不是论文在MoMaRT上的实验结果。

## 1. BBox-DP：最直接的“检测框 + DP”

《Choose What to Manipulate: Revealing Data Scaling Laws in Bounding-Box Guided Policies for Semantic Manipulation》，读取arXiv v2。原RGB、当前画框RGB、固定的带框首帧和本体状态共同条件化U-Net扩散策略；并非独立裁出一个ROI再追加高频坐标token。离线通过Label-UMI/SAM2得到标注，再训练上线YOLO。训练随机漏掉或错标当前帧框，首帧用于维持目标指代。论文允许用不同颜色区分多个对象角色，但这不构成门→篮架→碗自主切换的验证。其部分评价是人工阶段归一化得分，不能直接与我们的二值总成功率并列。

证据：[论文III-B、III-C、V节](https://arxiv.org/html/2602.11885v2)。值得借鉴的是原图保留和检测误差增强；固定首帧是历史信息，作为No-History对照时需单独标记。

## 2. Point-VLA：框是用户意图接口

《Point What You Mean: Visually Grounded Instruction Policy》。模型接收当前多视角图像、文字及带框首帧。在线可由用户GUI画框，也可由MLLM根据人的指向生成框；自动训练标注不意味着在线自动判断阶段。论文还混合训练纯文字与视觉指令，并做空间平移、局部CutMix增强。表3中，扰动蛋托抓取的画框、文字坐标、目标mask分别为80%、37%、10%。这些是论文特定任务结果；文字坐标也不等同于我们的位置embedding，mask去背景更不等同于我们“全景+额外ROI”的输入。

证据：[论文3.2–3.4、4.5及表3](https://arxiv.org/html/2512.18933v1)。它支持研究提示形式与上下文保留，不能直接证明我们的坐标分支有害，也不能解决无人指框时的目标选择。

## 3. VP-VLA：显式解决“何时换目标”

《VP-VLA: Visual Prompting as an Interface for Vision-Language-Action Models》。上层Qwen3-VL判断子任务、对象和目标位置，SAM3生成空间提示；下层同时接收原观测与提示图。其事件触发器采用夹爪开合变化，更新子目标；训练还有关键帧坐标预测辅助损失，失败标注回合被过滤。这里额外增加了规划、定位和提示对齐训练，不是只把检测框接进原策略。

证据：[论文3.2–4.1](https://arxiv.org/html/2603.22003v1)、[官方代码](https://github.com/JIA-Lab-research/VP-VLA)。对MoMaRT的推论：夹爪变化只能作为触发线索；拉门未成功、篮架卡住或抓取重试时，仍需判断实际状态。若复用此路线，必须单测阶段误判与失败恢复。

## 4. Scene Graphs：多对象集合与跨帧记忆进入DP

《Expanding Spatial and Temporal Context for Robotic Imitation Learning With Scene Graphs》。先由LLM列出任务相关实体，再检测/分割、跟踪；节点包含DINOv2外观、可见时的2D框及世界系3D中心，与场景根节点共同条件化扩散策略。对象离开视野时清空2D框、保留3D中心，假设未再观察期间不移动。实机64回合中作者报告4次分割失败，故并未假设感知完美。

证据：[作者论文3.1–3.2、Failure Analysis及Runtime](https://pengqinhe.github.io/Scene-Graph-CVPR2026-website/assets/paper.pdf)、[项目页](https://pengqinhe.github.io/Scene-Graph-CVPR2026-website/)。方法段写逐步并行新检测，运行统计却写每回合只调用一次GroundingDINO；[当前官方仓库](https://github.com/jianingq/scene-graph-robot)列出Cutie依赖，而论文写XMem。复现需确认版本，不能拼接成一套确定实现。该路线覆盖多个相关对象，未要求先选唯一接触目标，但需要标定/深度及记忆，不能作为同条件No-History结果。

## 5. VIMA：ROI与坐标是每个对象的表示

《VIMA: Robot Manipulation with Multimodal Prompts》。使用领域微调的Mask R-CNN拆出多个对象；归一化框由MLP编码，裁图补成方形后缩放、用ViT编码，两路融合成对象token。策略结合多模态提示和交互历史生成动作。检测器训练偏向高召回，策略训练加入随机伪对象；另以真值框作为oracle对照。它提供了“多个ROI+位置共同输入”的先例，而非始终只强调一个固定目标。其桌面任务与高层动作接口不能直接代表MoMaRT的连续移动操作。

证据：[原论文第4节、C.2/C.4和D附录](https://proceedings.mlr.press/v202/jiang23b/jiang23b.pdf)、[官方代码](https://github.com/vimalabs/VIMA)。应借鉴对象集合及感知误差处理，而不是把使用ROI本身当作效果保证。

## 6. VIOLA：最接近当前ROI和位置分支的结构参照

《VIOLA: Imitation Learning for Vision-Based Manipulation with Object Proposal Priors》。预训练RPN取多个候选框，在整图ResNet特征图上做ROIAlign，再加框角点的正弦位置编码。它同时保留全局、腕部相机和本体特征，并组合当前及过去共10步观测；Transformer通过动作监督学习利用相关区域，动作头为GMM。训练有随机擦除、框位置小扰动。其咖啡任务分析涉及杯、机械手及咖啡机，适合作为多接触对象参考。

证据：[论文3.2–3.3、图5及附录A](https://arxiv.org/pdf/2210.11339)、[官方代码](https://github.com/UT-Austin-RPL/VIOLA)。这提示我们可以研究多候选与特征图ROIAlign，但两者应分开消融；它没有证明单独替换ROIAlign就能解决降分，也没有要求每个候选必须对应完美实例。

## 7. Oat-VLA：对象之外，还保留机械手附近细节

《Focusing on What Matters: Object-Agent-centric Tokenization for Vision Language Action Models》。FT-Dinosaur产生对象mask，聚合已有图像patch为7个对象token；再用专门训练的夹爪检测器选取周围3×3 patch，共9个局部token。对象分支不是给每个对象检测框后裁图。论文表2中，仅对象token为61.3%，加入夹爪局部信息的平均池化完整方案为77.1%（LIBERO消融平均）。这说明该设置下接触附近细节很重要，不能推出夹爪周围一定包含MoMaRT当前目标。

证据：[论文3节、表2、附录7.1](https://arxiv.org/html/2509.23655v1)。对我们可检验“手—把手/碗接触区域”的价值，避免只保留物体外观；但检测器与策略仍需适配本环境。

## 8. ObjectDiff：训练时降低对检测提示的依赖

《ObjectDiff: An object-centric diffusion policy with modality-specific conditioning for robot manipulation》，Knowledge-Based Systems 346，116124。出版方摘要与引言描述DINO检测引导的对象表示，以及Detector-Free Guidance：训练随机省略检测生成的空间注意力，保留RGBD自身定位能力。它还同时修改指令条件化与训练目标，不能把整套收益归因于检测框。

证据：[出版方公开摘要/引言](https://www.sciencedirect.com/science/article/abs/pii/S0950705126008506)。本次未取得完整方法正文和官方实现，故不填写具体对象数、ROI算子、在线切换规则或dropout比例；仅将检测条件随机撤去作为有来源的设计线索。

## 对当前MoMaRT设计的修正判断

目前已核验实现是固定红碗检测、单碗ROI与位置token，全景保留。已有探针显示ROI扰动影响动作，但尚未证明固定碗提示是6个百分点下降的主因。文献也不能替代这个因果验证。[本地诊断报告](momart_token_diagnosis_20261029.md)及[飞书原报告](https://feishu.cn/docx/GU6adqL59oGoNqxyCMOcsy7rn94)保留原始证据。

第一，需明确框的语义。如果把它定义为“当前操作目标”，就需要类似VP-VLA的在线子目标生成；如果把它定义为“场景对象”，则可参考VIOLA/VIMA，同时表示多对象，让策略学习关系。我们现在的碗token更接近固定对象观测，不能假定网络一定把它理解为当前目标，也不能假定网络总能学会忽略它。

第二，“目标覆盖”与“表示形式”要分开。覆盖门、篮架、碗，不等于必须全部变成紧裁图；ROIAlign、mask池化、原图画框是不同选择。看过文献后，我更建议先验证小规模多候选的可行性，并保留全景；这仍是待实验建议，不是已经选定的新模型。

第三，在线可靠性通常由多个环节共同获得：领域检测适配、多个候选避免过早排除目标、跟踪/历史、检测误差增强，或明确的规划器/人工指代。不能将“检测器找到多个物体”直接等同于“知道当前操作对象”。

## 建议的验证顺序（尚未执行）

1. 在现有十任务视频中，检查门/把手、篮架/拉手、碗及机械手区域的候选覆盖、遮挡、漏检与延迟。训练示范用于工程排查，独立回合用于在线泛化验证。人工阶段标注只用于分组分析，不作为正式评测输入。
2. 首轮策略比较保留当前观测窗口与DP主体：全景基线、固定碗token、多候选对象token；固定训练数据、预算、学习率计划及评测seed。多候选数量、padding/mask和对象采样需明确。既有只画框结果仍保留，避免与新输入混称。
3. 在同一个多候选设计上单独比较检测误差增强，例如漏检、框扰动和伪对象；不要同时换backbone、ROI算子、历史长度与训练损失，否则难以解释变化。
4. 若失败仍集中于阶段歧义，再单列有历史的对象记忆，或带阶段规划器的实验。首帧锚定、10步历史、场景图记忆都会改变No-History条件。

最终验收仍只用同协议下的完整任务总成功率，并与既有No-History比较。论文中的不同任务成功率、阶段分数和检测准确率，不汇入MoMaRT同一张成功率柱状图。本次是设计调研，未启动训练、未新增在线成功率。
