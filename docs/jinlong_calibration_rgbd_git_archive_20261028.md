# 2026-10-28 最新手眼标定与三型号真实RGB-D Git归档

按用户要求，已将最新23组右臂手眼标定及160／132／100真实彩图、原始16位深度图和深度预览上传至[机械臂项目GitHub资料目录](https://github.com/thedaosheng/jinlong-motor-arm-control/tree/da9392e17e6af551d18607231244b56fccd16d82/reference/2026-10-28-calibration-rgbd)，分支 `data/calibration-rgbd-20261028`，提交 `da9392e17e6af551d18607231244b56fccd16d82`。资料约34.05 MB，直接存放于Git，可克隆读取；附内参、同帧采集元数据、来源清单和SHA256校验文件。

本分支从远端main建立，只提交本次资料及入口，未推送本地主分支另外85个开发提交，也没有更新工控机程序或控制机器人。本次用户明确要求上传上述采集文件，构成仓库一般不收录采集数据规则的限定例外。

## 标定

工控机活动文件重新回读，与入库文件SHA256一致：`76d7613ccb761512b1609e86e53b7269cac3b2cce97c971d6a82c70c2331bd41`。会话 `20261028_150650_65e404e3c8ba`，23组右臂样本，ANDREFF；内部残差0.296442mm／0.074505°。同时入库原始 `handeye_result.json`，其SHA与活动标定 `source_sha256` 一致，保留每样本残差和左右臂／世界变换。不是新增标定或独立精度验收。

## 三型同帧数据

| 型号 | 完整帧采集时间（北京时间） | 当次深度窗口 | 盒口中心相机Z | 文件入口 |
|---|---|---|---|---|
| JM160 | 2026-10-28 15:33:09.092 | 350–800mm | 387.783mm | [160彩图／深度／内参](https://github.com/thedaosheng/jinlong-motor-arm-control/tree/da9392e17e6af551d18607231244b56fccd16d82/reference/2026-10-28-calibration-rgbd/JM160) |
| JL132 | 2026-10-28 15:28:55.284 | 400–800mm | 473.734mm | [132彩图／深度／内参](https://github.com/thedaosheng/jinlong-motor-arm-control/tree/da9392e17e6af551d18607231244b56fccd16d82/reference/2026-10-28-calibration-rgbd/JL132) |
| JL100 | 2026-10-28 15:42:24.566 | 350–800mm | 550.934mm | [100彩图／深度／内参](https://github.com/thedaosheng/jinlong-motor-arm-control/tree/da9392e17e6af551d18607231244b56fccd16d82/reference/2026-10-28-calibration-rgbd/JL100) |

三型均为3200×1944，RGB和深度包内帧号匹配；原始PNG直接复制，未降采样或重新编码。在工控机将三张原始深度PNG分别与对应NPY逐像素比较，均完全相等，dtype为uint16，因此Git不重复保存NPY。原始深度每计数约0.025mm，以各自内参精确值为准；0为无效。伪彩图仅用于预览，不能用于测量。

160选用最新保存完整RGB-D的350mm诊断帧；稍晚15:38:58网页复验只有摘要和标注，不将其冒充原始图像。132网页成功任务和补存完整帧是不同采集。100于15:42:24已保存新标定后的真实完整帧，故早期报告“100尚待采集”是旧阶段状态，应予更正。项目当前统一350–800mm，132归档忠实保留当时400mm下限。

## 原图与深度预览

以下为上述同帧文件；可下载原始16位深度的Git入口见表格。

![160真实彩图](../outputs/diagnostics/new-calibration-common-waist-20261028/live-captures/JM160/depth350-diagnostic/fusion/trigger_000001/01_rgb_original.png)

![160同帧深度预览](../outputs/diagnostics/new-calibration-common-waist-20261028/live-captures/JM160/depth350-diagnostic/fusion/trigger_000001/04_depth_pseudo_auto.png)

![132真实彩图](../outputs/diagnostics/new-calibration-common-waist-20261028/live-captures/JL132/full-evidence/fusion/trigger_000001/01_rgb_original.png)

![132同帧深度预览](../outputs/diagnostics/new-calibration-common-waist-20261028/live-captures/JL132/full-evidence/fusion/trigger_000001/04_depth_pseudo_auto.png)

![100真实彩图](../outputs/diagnostics/new-calibration-common-waist-20261028/live-captures/JL100/full-evidence/fusion/trigger_000001/01_rgb_original.png)

![100同帧深度预览](../outputs/diagnostics/new-calibration-common-waist-20261028/live-captures/JL100/full-evidence/fusion/trigger_000001/04_depth_pseudo_auto.png)

## 校验与使用边界

已核对工控机源文件SHA、所有入库JSON解析、9张PNG解码、图像尺寸／内参一致、深度位宽及包内流帧号。`sha256sum -c SHA256SUMS`全部通过，远端分支已成功推送。

活动标定 `base_T_camera` 指向标定参考腰姿下的左臂SDK基座。CLI `capture.json` 中 `base_T_box` 不能直接代表当前腰姿目标；须用相机系观测、固定世界相机与当次腰姿换算。此次只归档数据，不代表三型实物插接、电测或完整运动流程已通过。

[飞书归档报告](https://feishu.cn/docx/I2GAdun3nofskUxAOjWcxVwjnJO#doxcnV5chnaDwDLyhdGp1viUmdc)已追加并回读核对：新增60块，33个文字块、1张表和6张3200×1944图像一致，全部旧正文保持不变。远端资料分支SHA已再次核对为`da9392e17e6af551d18607231244b56fccd16d82`。
