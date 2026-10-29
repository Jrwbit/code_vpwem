# MoMaRT 原始全画面与 ROI：同步3倍速视频｜2026-10-29

飞书视频附件：https://feishu.cn/docx/F64rdnIUJoPkrNxCXGScLb1ynEN

用户确认需要同一段视频的原始全画面与检测框内ROI，分别提供，两份均加速3倍。

附件：original_3x.mp4 为未画红框的原始主RGB；roi_3x.mp4 为同一帧按检测框裁剪后的ROI。

来源为 table_cleanup_to_sink_expert 的 demo_0 训练演示，不是新模型rollout。原始566帧，按既有演示20fps为28.30秒；两份输出均保留566帧，以60fps播放，时长9.433秒，逐帧对应，无音轨。

ROI严格沿用训练输入规则：归一化xyxy裁至[0,1]，左上floor、右下ceil，从未画框原图裁剪，双线性缩放64×64（align_corners=False）。有效框473帧；其余93帧ROI显示黑色，对应模型缺框时的零输入。此处是离线训练框，不冒充在线combined_blend跟踪框。

原图原生120×90，ROI原生64×64。为便于观看，均以最近邻放大8倍，分别为960×720和512×512；视频编码有损，未增加图像细节。

数据源：BHR /home/jrw/vpwem_data/momart_zarr/table_cleanup_to_sink_expert.zarr 中 data/rgb 的[0:566]，框源 /home/jrw/vpwem_data/momart_bbox_gdino_red_v1/table_cleanup_to_sink_expert.bbox.zarr 中 boxes/valid 同帧范围。帧数及20fps依据 /home/jrw/vpwem_outputs/momart_bbox_videos_20261009/manifest.json。

制作脚本：tools/export_momart_roi_video_pair.py。原始数据、视频和ffprobe核验清单保存在Git忽略目录 artifacts/momart_roi_videos_20261029/；只提交脚本及本记录，不提交数据或视频。

验证：合成有效框/缺框视频通过；两份正式视频经ffprobe核验均为566帧、60fps、9.433333秒，并人工检查同一时刻的原图和ROI画面。飞书两份附件已回读文件块，并重新下载核对字节数及SHA256，与本地文件一致；正文已回读核对。

## 后续：十任务与黑屏说明

用户反馈旧视频看起来全黑后，逐帧解码确认：原图566帧都不是黑帧；ROI为473帧可见、93帧黑色，首36帧缺框（3倍速0.6秒）。这不能排除客户端播放问题。新版改用30fps H.264 baseline，明确显示缺框状态并提供原图/ROI同屏，补齐10任务。详见 docs/momart_roi_ten_tasks_20261029.md，飞书：https://feishu.cn/docx/QZ0rdrBJloHhqqxrJXFcISKinue 。旧视频保留作历史记录。
