# Repository working rules

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
