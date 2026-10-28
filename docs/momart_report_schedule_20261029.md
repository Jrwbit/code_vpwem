# MoMaRT 2026-10-29 08:00 汇总安排

用户要求：明早08:00将评测结果整理到新飞书文档，含此前实验对比。

已安排本机at一次性任务123，触发时间2026-10-29 08:00:00 Asia/Shanghai；atd运行中，atq已回读确认。届时启动整理，读取真实结果、核验后发布新文档，生成文档需要处理时间，不承诺08:00整已有最终文档。依赖本机届时开机且可连接BHR和飞书。

执行器：`tools/scheduled_reports/run_momart_20261029_0800.sh`；完整上下文及交付要求：`tools/scheduled_reports/momart_20261029_0800.prompt.md`。采用现有Codex登录和连接配置，不保存额外凭据。非交互模型调用已验证回复READY；bash -n通过。日志在 `/home/wjr/.local/state/momart-report-20261029/`，非Git运行产物。

范围：175k/200k/225k、250k原始评测、同seed100三轮、200/300两轮；此前150k带框、150k带框+ROI token、150k action4、普通No-History 1M及有证据的相关方法。统计总成功率、任务分项、同seed重复一致性与跨seed变化，明确可比性和训练预算差异。取消过的未完成旧轮次排除。

预期新文档标题：MoMaRT 评测总对比与250k重复性验证｜2026-10-29。定时任务发布后回读正文/表格/图片，将链接和报告提交Git。08:00仍未完成的项目按当时状态列出，不能伪报最终成功率。此时尚未生成结果文档。

调度依据：[Codex非交互执行官方文档](https://learn.chatgpt.com/docs/non-interactive-mode)，本机CLI参数已核对。
