# MoMaRT 十任务原图与ROI：两倍速视频｜2026-10-29

飞书视频及下载：https://feishu.cn/docx/CYdidYMf7oJKD3xjc2HcnRrjnFb

按最新要求，十个任务全部改为2倍速。每个任务提供原始全画面、ROI与左右同屏对照；本页附十个同屏视频及包含20份独立视频的分卷压缩包。

全部使用30fps、H.264 baseline、yuv420p、faststart，无音轨。片头明确显示2x，缺框显示NO VALID DETECTION。缺框时模型的ROI图像为零；展示版使用灰底提示，方便区分缺框和播放异常。改成两倍速不会补出原本缺失的检测框。

三倍速旧版的视频源本身并非全黑：此前单任务ROI中473帧有画面、93帧因缺框为黑色。客户端全黑的原因尚未确认；新版保留更兼容的编码和缺框文字提示。

同一套训练演示、同一组离线框，无新训练或模型rollout。原速按20fps，两倍速为每秒40个源帧；以30fps采样输出，片长为原速的一半，取整误差不超过1/30秒。原图和ROI逐帧对齐。

## 十个同屏视频
- table_cleanup_to_dishwasher_expert：17.50秒。
- table_cleanup_to_dishwasher_suboptimal：31.63秒。
- table_cleanup_to_sink_expert：14.17秒。
- table_cleanup_to_sink_suboptimal：26.40秒。
- table_setup_from_dishwasher_expert：9.17秒。
- table_setup_from_dishwasher_suboptimal：10.47秒。
- table_setup_from_dresser_expert：14.77秒。
- table_setup_from_dresser_suboptimal：15.37秒。
- unload_dishwasher_to_dresser_expert：18.30秒。
- unload_dishwasher_to_dresser_suboptimal：26.00秒。

## 原图和ROI独立视频下载
- original_roi_2x_part01.zip：table_cleanup_to_dishwasher_expert、table_cleanup_to_dishwasher_suboptimal、table_cleanup_to_sink_expert
- original_roi_2x_part02.zip：table_cleanup_to_sink_suboptimal、table_setup_from_dishwasher_expert、table_setup_from_dishwasher_suboptimal、table_setup_from_dresser_expert
- original_roi_2x_part03.zip：table_setup_from_dresser_suboptimal、unload_dishwasher_to_dresser_expert、unload_dishwasher_to_dresser_suboptimal

30份新MP4均已检查帧数、时长并完整解码。视频与核验记录保存在Git忽略目录artifacts/momart_roi_2x_20261029/；源帧沿用artifacts/momart_roi_ten_tasks_20261029/sources.zip。导出脚本为tools/export_momart_roi_ten_tasks.py，参数--speed 2。

飞书13份附件（10份同屏视频、3个独立视频压缩包）已回读文件块并逐份重新下载，字节数和SHA256全部匹配；正文已回读核对，旧三倍速页面已追加新版入口。

此前三倍速版：https://feishu.cn/docx/QZ0rdrBJloHhqqxrJXFcISKinue
