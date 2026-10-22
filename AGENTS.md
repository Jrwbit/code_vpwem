# Repository working rules

## 执行与确认原则

- 用户交代的任务直接执行；“帮我做”“弄一下”“能不能做”等表达均视为行动指令，不停留在建议、计划或询问是否继续。
- 默认自主推进任务，直到完成实现、必要验证和交付。
- 任务范围内的读文件、排查、修改代码、运行测试和 Git 提交直接执行，无需逐步确认。
- 普通信息缺失时，采用符合现有代码和上下文的合理默认值，交付时说明影响结果的假设。
- 只有缺失信息会实质改变结果且无法从现有上下文合理推断，或操作涉及未授权的不可逆变更、对外发送、额外费用时，才暂停询问；暂停前先完成不依赖该答案的已授权工作。
- 已获得的授权在本任务内持续有效，不重复确认；用户明确要求的飞书同步和 Git 提交按既定规则直接执行。
- 遇到障碍先尝试合理替代方案；确实无法继续时，报告具体阻塞和已尝试的方法，不虚报完成。
- 验证范围与改动风险相称。相关检查通过后，不因假想风险反复检查或扩大测试。
- 不添加与当前任务无关的兜底、兼容层、异常吞噬或防御性抽象。
- 回答先给结果，只说明影响决策的风险和限制，不附加无关的免责声明或确认流程。

## Reports must be synced to Feishu

- 用户要求：任务报告、实验记录和排查结论默认写入飞书，不能只保存在本地或只提供 Git 提交号。
- 机械臂检测相关报告优先追加到既有全流程报告：https://feishu.cn/docx/I2GAdun3nofskUxAOjWcxVwjnJO 。保留原正文，补充新结论及对旧判断的更正；其他主题沿用对应文档或创建新文档。
- 同步正文、表格和必要图像；附件按任务需要上传，不将本地文件链接冒充可下载的飞书附件。
- 写入后回读核对正文和图像，交付时提供飞书链接。若同步失败，明确说明失败，不能宣称已经写入。
- 本地源报告及相关代码仍按下方规则记录到 Git；本地留档不替代飞书同步。

## Git is mandatory

- Every code, configuration, documentation, test, script, or module change made for a task must be recorded in Git before the task is handed back.
- Create an atomic commit after each independently meaningful module or coherent change set; do not leave completed work only in the working tree.
- Stage only files belonging to the current change. Preserve unrelated pre-existing changes and never use a blanket `git add -A` in a dirty worktree.
- Inspect `git status` before staging and again after committing, and report the resulting commit hash.
- Never commit secrets, credentials, private keys, datasets, model weights, caches, generated runtime output, or other large artifacts. Add an appropriate ignore rule when needed.
- Run relevant checks before committing whenever practical. If a check cannot be run or fails for a pre-existing reason, record that clearly in the handoff.

## 机器人检测项目长期上下文

- 继续金龙／台州 160、132、100 电机检测任务前，先读 `docs/jinlong_project_context.md`，再读其中指向的最新报告与证据；日期较新的现场记录优先，不把历史快照当实时状态。
- 目标是三型号从扫码、定位、工装对位、轻触／补偿压入、电测到撤离返回的完整流程；终点有解、机械往返成功、电测验收是不同结论。
- 更新项目关键结论时同步维护该上下文入口，并按既有规则提交 Git、追加飞书，保留历史证据和更正依据。
