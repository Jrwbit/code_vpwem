# 当前接线盒识别到悬停仿真联调（2026-10-17）

已完成现场“新标定 → 新采集 → 型号工装 → 悬停目标 → 碰撞仿真”的软件链路，
并打开本次识别对应的候选轨迹视频。没有执行真机运动；真实自动悬停尚未完成。

## 已部署的识别与标定接口

源码位于开发机 `/home/wjr/code/jinlong-motor-arm-control`，新增
`vision/app/hk_pointcloud_registration/robot/rokae_hover.py`。
控制器和 HTTP 接口支持：

- `POST /api/robot/calibration/import`：本机控制会话及活动文件 SHA 校验，
  核对保存样本的手臂／控制器、样本数量、腰部一致性、内部残差，
  通过本机 URDF 独立复算右基座到左基座以及整机基座变换，再原子替换活动标定。
  旧文件自动存档，旧观测和计划失效，独立验收始终重新置为 false。
- `POST /api/robot/hover/preview`：只接受当前任务、600 秒内的新识别结果、
  相同标定摘要、有效观测代次、新鲜且停稳下电的整机状态、固定标定腰姿态、
  正确的型号与工装，以及一致的相机到基座矩阵。
  输出为非可执行目标，`executable=false`、`controller_writes=false`。
- 页面新增“悬停距离（毫米）”及“预览工装悬停目标”。距离允许 60～300 mm，
  该范围沿用既有离线规划所搜索的非接触高度；本次界面初值为 110 mm。
  数值合法不表示路径合格，仍必须完成全路径仿真和现场验证。

软件源提交为 `db2aba3`、`6ed6c7a`；部署至原工控机视觉仓库后，
对应提交为 `33546d1`、`c068531`。运行版本最终核对为
`c0685310a68b705fb9f007672505699eb956b0b9`，`dirty=false`，142 个源码文件。
本次没有向 GitHub 推送。

## 现场参数与本次观测

应用右臂 7 样本会话 `20261016_210737_3233a5f3f80f`，替换原 10 月 14 日活动标定。
左右臂名义安装关系的转换检查通过；当前腰部与新标定基准相符。
内部离散约 1.40 mm／0.277°，没有据此宣称实际定位精度通过。

工位工具矩阵补上文档明确的 `flange_T_tcp=Rz(-120°)`。
接入现场既有 YAML 中的 JL100 −18 mm、JL132 −22 mm 高度修正，
以及 JM160 盒局部 X −2 mm 修正。选中工装与本次 JL132 识别一致。
原 TCP、工装、路径、工作空间及控制验收未通过的标记仍为 false，
没有填写虚构的工作空间、初始位置或允许误差。

恢复原 WorkingMode 4 相机常驻模式及原有曝光／检测参数后，重新真实采集。
最终输入任务为 `20261017_114547_4d888c`，型号 132，
`observation_context.valid=true`，绑定新标定的 SHA。
没有将旧识别结果仅乘上新矩阵冒充重新采集。

## 本次悬停规划结果

离线使用现有 `jinlong-motor-inspection/sim/helios_insert` 的转盘、插座、
接线柱、桌面、相机禁入区、腕部联合限位及左侧调姿通路实现。
只更新新观测中的盒／电机位姿和相机位姿，桌面保持世界固定；
头部及右臂使用本次反馈，保留右手标定板的保守模型。

150 mm 的初始悬停搜索没有找到合法通路；按项目原有 JL132 悬停方案采用 110 mm，
并在新盒位下重新计算。候选需要先收臂，再调整腰部到模型角
`[-33.78, 35.11, 6.84, -46.09]°`，然后在左侧空旷区域调姿，经直线通路到盒外悬停。
这四个值是仿真候选，未下发到控制器。

最终新采集对应方案共 10 个运动小步；视频约 41 秒（1238 帧，30 fps）。
模型中稠密检查的最小有效间隙为 25.5 mm；直线通路约 33 cm，
按关节直线近似执行时，末端偏离上限约 2.94 mm／0.69°。
大幅调姿没有进入模型规定的禁调区域，腕部联合限位检查通过。
以上基于名义运动学和历史场景尺寸，不是实测碰撞或精度验收。

由于候选改变腰部姿态，另生成 `working_calibration_candidate.json`：
保持世界相机位姿不变，通过 URDF 推导工作腰姿态下的左臂基座相机变换。
它明确标记为名义重投影、未应用、独立验证 false；没有替换当前活动标定。
因此不能直接绕过固定腰部检查运行这条候选。

## 工控机上的查看方式

本次视频、终点图片、报告和名义标定候选存放于：

```text
/home/bit-bhr/jinlong-fixture-replay/20261017-current-hover/
```

桌面新增“本次识别悬停仿真（未执行）”，只播放 `hover.mp4`。
检测页面已在 Firefox 重新打开，地址仍为 `http://127.0.0.1:8765/`。
原浏览器实例未刷新打开请求，后改用本任务专用 Firefox profile 和原生 Wayland 打开两页；
回放启动器也使用该 profile，避免复用旧相机预览窗口。
轨迹没有复制到 HMI 的执行计划目录，也没有调用上电、拖动或运动接口。

## 可重复使用的开发机入口

`tools/jinlong_hmi/capture_hover_bundle.py` 从工控机现有本机 HTTP 服务取得
新任务、活动标定、工装及真实起点，输出非可执行仿真输入；保存前移除 HMI CSRF 凭据。
`--capture` 才触发新采集，`--import-session` 才显式导入保存标定。
该脚本不调用任何机器人运动或上下电接口。

`plan_vision_hover.py` 接入原仿真源码，重新计算并稠密检查候选路径，
失败则只输出阻塞诊断；成功输出视频及明确未授权执行的计划。
历史计划只可用作逆解初值，不复用其轨迹或检查结果。
依赖在独立 `/tmp/jinlong-hover-env` 中，版本见 `requirements-hover.txt`。

```bash
python3 tools/jinlong_hmi/capture_hover_bundle.py \
  --ssh-host jinlong-ipc --capture --model 132 --hover-mm 110 \
  --out /tmp/new-hover/bundle.json

MUJOCO_GL=egl /tmp/jinlong-hover-env/bin/python \
  tools/jinlong_hmi/plan_vision_hover.py \
  --sim-root /path/to/jinlong-motor-inspection \
  --vision-root /path/to/jinlong-motor-arm-control/vision \
  --bundle /tmp/new-hover/bundle.json --scene /path/to/site-scene.json \
  --out /tmp/new-hover/simulation --work-waist-deg -33.78 35.11 6.84 -46.09
```

这组工作腰参数只是本次候选，不能直接作为其他工位的生产参数。
相机常驻模式未启动时可显式加 `--ensure-resident`；服务重启后可用
`--params-json` 传入原有相机／检测参数，避免默默换用默认参数。

## 验证、备份及剩余工作

- 开发机视觉完整隔离回归 281 项通过，其中 3 项缺 Node 跳过。
  后续悬停距离变更的 5 项针对测试在开发机及工控机均通过。
- 新增测试验证左右臂转换、标定导入授权／版本冲突／旧件备份／不继承独立验收、
  三工装姿态与补偿次序、旧结果拒绝、非接触距离范围。
- 腰部重投影测试通过，验证世界相机不变且不会继承旧验收。
- Python 编译、Node 22 对新增前端逻辑的语法检查、启动脚本语法和桌面入口验证通过。
- 本次视频成功解码；视频／图片／报告／候选四个文件传输后 SHA 全部一致。
- 最终整机仍为新鲜反馈、下电、手动、空闲，HMI busy=false。
  检测端 `motion_enabled=false`，仍明确报告独立定位、工具、工作空间及路径验收缺失。

工控机旧源码、Git 元数据、旧标定和配置备份在
`~/.local/state/jinlong-hmi/deployments/20261017-vision-hover/`；
标定导入与工位保存也分别留下 API 自带历史记录。
便携 Git 的自动 gc 辅助程序仍缺失，提交本身成功并已复读核对。

下一步需要现场提供至少三个未参与标定的非共线验证点及允许误差，
确认工作腰姿态下的定位、实际 TCP／工装、工作空间和完整路径，之后才能启用真实悬停。
已向用户询问是否有这些实测数据。没有编造测量点或把确认标记改成 true。
视频、图像、标定、识别结果、环境、日志和运行数据均不提交 Git。
