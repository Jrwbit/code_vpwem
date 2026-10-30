# MoMaRT多对象缓存：耗时剖析与已处理帧质量审计

日期：2026-10-30（北京时间）。飞书：https://feishu.cn/docx/LHqzd8Cjpo9O0RxgR2DcY1w0nIf

**结论：当前多对象感知既慢，又存在明显的类别误识别，不能作为已达到95%可靠性的训练输入。已停止预处理和其后自动训练链，保留20928帧已提交缓存，正式A1训练尚未开始。** 先前先启动全量缓存、再补质量检查的顺序不合适；后续应先通过小样本对象级质量验收，再投入全量计算。

这次不改变策略架构、不重训检测器。工作为已处理缓存的只读抽查、分段计时、错误定位及状态归档。

**为什么慢：当前实现重复处理六份相同图像**

每张原图仅90×120，却为了GroundingDINO被放大为约600×800；单份像素面积约扩大44.4倍。六个类别提示对应六份相同图像，在一次6样本batch前向中重复做图像/文本处理和视觉计算，没有复用图像骨干或固定提示的文本编码。程序随后串行处理下一帧；没有跨帧组成吞吐批次。

这是每帧六个图像—文本样本，不是六个Python forward调用。全量654460帧对应约392.7万个此类样本，原FP32管线平均约2.46帧/秒，因此产生约三天的感知准备成本。没有证据把主要时间归因于memory读取或磁盘。

对同一已缓存帧10133，在物理GPU5运行3次，剔除首轮模型/算子热身，另外2次加同步分段计时：

| 环节 | 暖机后实测 |
|---|---|
| CPU图像/文本processor | 198.90–209.77毫秒/帧 |
| GPU GroundingDINO前向，六提示batch | 228.57–233.16毫秒/帧 |
| 原始框后处理 | 约2.50毫秒/帧 |
| 冻结对象ResNet layer2 | 约1.01毫秒/帧 |
| 完整感知调用 | 453.83–459.51毫秒/帧 |

这个单帧profile用于定位瓶颈，不是整个数据集的平均值或延迟分位数。额外独立测得128帧的原读取路径共0.1666秒，约1.30毫秒/帧；用现有read_frames批读只需0.0115秒/128帧。批读值得改进，但仅此不能把当前0.4秒级检测降成快速预处理。

**如何抽样与复核**

冻结抽样总体为首个数据集 `unload_dishwasher_to_dresser_suboptimal` 已处理的前20608帧，用NumPy default_rng、seed20261030、无放回简单随机抽36个anchor。读取SQLite中已保存的231个输出框，没有重新挑高置信样本，也没有把后续帧用于修正结果。图像经training_replay按同一训练帧映射读取，并核对episode和源帧映射。审计不向原缓存写入标注或更正。

审查材料为原始90×120RGB与单类别框并排放大图。此处的逐框目视复核由本助手完成，没有独立人工复核，也没有为所有对象建立像素级/框级GT，因此不报告正式precision/recall/mAP、不估计总体置信区间。

| 类别 | 本次样本输出框数 | 本轮能够支持的判断 |
|---|---:|---|
| 红碗 | 22 | 目视均落在可见红碗附近，但部分遮挡框包含夹爪；未按GT计算IoU，不能写成100%定位正确 |
| 洗碗机门 | 38 | 存在把碗、机械臂、柜体和整幅图像标成门的情况；未完成所有框定量标注 |
| 洗碗机门把手 | 33 | 逐一复核33框，均为明显错类或严重错位：落在红碗、机械臂/夹爪或过大场景区域 |
| 洗碗机篮架 | 43 | 存在整图、柜体或红碗框，部分真实篮架近景需GT进一步判定 |
| 抽屉 | 56 | 存在整图、机器人和多个柜面混框；未完成逐实例GT |
| 抽屉把手 | 39 | 存在柜体大框、机械臂和碗等明显问题，且返回phrase与所赋类别不一致 |

只计算已经明确标错的33个门把手框，其余198个框全部乐观算作正确，样本输出正确比例上限为 `(231-33)/231=85.71%`。这是假设目视错误判定成立时的保守排除证据，**不是测出了总体准确率85.71%，也不是正式GT precision**。这一随机前缀样本已经不支持“当前版本95%以上可靠”的判断；其他任务/类别仍需独立标注评测。

客观几何统计辅助解释现象：68/231框覆盖至少90%图像面积；13/33门把手框与同帧红碗框IoU≥0.7。几何重叠本身不是GT错误标签，应结合原图判断。

**三个可回看的错误案例**

左列原图，右列为缓存中的“洗碗机把手”框。帧10133把红碗框成把手，置信分数约0.610；帧6181框住机械臂/夹爪；帧2188几乎框住整幅图像。放大使用nearest，不补造原图细节。

![缓存错误案例：红碗、机械臂和全图被标为把手](../artifacts/momart_perception_audit_20261030/error_examples.png)

以下为本次33个门把手框的完整逐项审查图，每项左边原图、右边预测框。图片标签保留frame和prediction index，可定位SQLite记录。

![全部33个门把手输出：原图与预测框配对](../artifacts/momart_perception_audit_20261030/class_2.png)

红碗输出仅作对照，不能据此推断部件检测同样可靠。

![本次22个红碗输出：只作语义位置对照](../artifacts/momart_perception_audit_20261030/class_0.png)

**模型局限之外，明确存在的实现缺口**

`momart_object_memory/perception.py`按prompt的序号直接赋class_id，仅保留boxes和scores，没有核验检测器实际返回的phrase。对帧10133回读原始后处理结果：

- 红碗提示返回 `a red bowl`，框在红碗上，分数0.812。
- 洗碗机门提示返回 `a dishwasher door`，仍框在同一个红碗上，分数0.426。
- 洗碗机门把手提示返回 `a dishwasher door handle`，仍框在红碗上，分数0.610。这属于实际grounding失败，仅补phrase检查不能解决。
- 抽屉把手提示实际返回多个 `a drawer`，代码却直接赋成drawer_handle。这是类别映射/验证缺口，不应当作已正确识别把手。
- 篮架提示还返回残缺短语 `aer`，该结果同样没有被phrase一致性检查排除。

此外，当前“每类优先保留一个”的容量选择会保留已过低分阈值但不相关的类别候选；它不会提高识别正确率。0.25分数阈值也不等于25%或95%的实际准确率。旧combined_blend的高precision只覆盖红碗，不能赋给这些部件。

**已采取的处理与当前状态**

只对核验命令行确属本任务、且stage=perception的PID2858696发送SIGTERM，避免继续扩大低质量缓存和自动进入训练。最后一次完整提交为20928帧；SQLite未提交尾部可由事务恢复，原数据和已提交缓存保留。状态文件为 `stopped_for_quality_audit`，另写 `quality_audit_stop.json`，没有修改原基线训练或其他GPU任务。

原先11月2日完成/完成后自动训练的预计已不再是当前安排；任务已停止在质量审计节点，正式A1没有开始。模型/数据不能在未经修正与验收后直接接着使用这批缓存训练。

下一次实施应先固定对象与部件标注标准，修正phrase到类别的验证，并用独立小集测试部件precision、recall和框定位。若通用grounding仍严重混淆，应考虑领域检测器或父物体约束下的部件定位。速度优化可在质量成立后比较单图联合提示、图像/文本编码复用、跨帧batch、缓存固定文本及混合精度；这些都会改变感知协议或数值，需要同样用于A1/A2/A3并重新验证，不能只给某一组换检测器。

**复现与归档**

`tools/audit_momart_cached_perception.py`提供只读sample和score命令。sample固定总体/seed，导出原图、类别配对图和预测索引；score只接受外部复核记录，不用置信度冒充正确率，并将未复核样本全部按正确计算上述上限。

```bash
python tools/audit_momart_cached_perception.py sample \
  --source /home/jrw/vpwem_data/momart_zarr/unload_dishwasher_to_dresser_suboptimal.zarr \
  --database /home/jrw/vpwem_data/momart_objects_causal_bounded_v2/unload_dishwasher_to_dresser_suboptimal.perception.sqlite \
  --population 20608 --samples 36 --seed 20261030 --output AUDIT_DIR
python tools/audit_momart_cached_perception.py score \
  --sample AUDIT_DIR/sample.json --review AUDIT_DIR/visual_review.json --output AUDIT_DIR/summary.json
```

抽样导出已在服务器实际运行，复现36帧/231框；score实际检查33个明确错误引用，得到0.857142857的条件上限；编译及git diff检查通过。逐帧原图、预测、复核JSON和运行输出保存在artifacts及服务器审计目录，不作为数据集提交Git。飞书上传真实审计图，不能将本地图片路径当飞书附件链接。原[启动记录](https://feishu.cn/docx/E7uqdKhYKo4Cd3x8sYQcldmtnue)保留并追加停止状态。
