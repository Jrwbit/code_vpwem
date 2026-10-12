# 珞石与海康离线手眼标定准备工具

该工具只读取现有控制程序导出的本地文件，计算标定结果。不导入机器人或海康 SDK，不访问网络，不修改设备参数，也不启动、停止任何服务。现有控制程序仍是设备的唯一控制方。

## 工控机运行

已验证的解释器：`/data/rhz/conda-envs/ur5_data_collection/bin/python`。此工具不需要海康动态库路径。不要为运行它改动正在控制设备的环境。

在工具目录执行：

```bash
/data/rhz/conda-envs/ur5_data_collection/bin/python calibrate.py check-environment
/data/rhz/conda-envs/ur5_data_collection/bin/python -B -m unittest -v test_calibrate.py
/data/rhz/conda-envs/ur5_data_collection/bin/python calibrate.py solve --input datasets/session/dataset.json --output results/session/report.json
```

退出码 0 表示通过所配置的内部质量检查；2 表示数据、求解或质量检查失败。报告仅保存候选结果，不自动应用到现有控制系统。数据、图像、报告均不提交 Git。

## 现有控制程序需要导出的数据

复制 `dataset.template.json` 至 `datasets/session/dataset.json`，填写真实内参、标定板参数和样本。模板中的内参是占位值，板参数和阈值是示例，不可直接当作现场参数。

- `eye-to-hand`：相机固定在外部，板刚性固定在机器人末端。方程为 `base_T_tcp @ tcp_T_board = base_T_camera @ camera_T_board`。
- `eye-in-hand`：相机刚性固定在末端，板固定在外部。方程为 `base_T_tcp @ tcp_T_camera @ camera_T_board = base_T_board`。
- `A_T_B` 表示把 B 系坐标变换到 A 系。所有平移单位为米，旋转为 3×3 矩阵，禁止把欧拉角直接当成 UR 旋转向量。珞石接口的毫米／角度数据由现有程序按其实际轴顺序、单位转换后导出。
- 使用真实反馈的 `base_T_tcp`，不是命令目标位姿。所有样本使用同一个 TCP 定义、相机安装和内参。相机曝光时刻及机器人位姿时刻使用同一个时钟，以秒表示。
- 图像必须是原始 RGB 图像，不能使用缩放、裁剪或检测叠加图；宽高必须匹配内参。机器人应在曝光及位姿读取期间保持静止。
- 先等待机器人停稳，再由现有控制程序配对导出图像与位姿。建议采集 15–25 组，在视野内改变位置并绕至少两个不同轴旋转。工具最低接收 10 组，拒绝单轴旋转退化数据。

每个样本的格式如下（矩阵必须替换为真实反馈；示例不代表有效标定姿态）：

```json
{
  "id": "001",
  "image": "images/001.png",
  "image_timestamp_s": 1000.0,
  "pose_timestamp_s": 1000.01,
  "robot_stationary": true,
  "base_T_tcp": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
}
```

图像路径相对于 dataset.json，必须位于数据目录内。也可以用现有程序已经检测出的 `camera_T_board`（4×4 矩阵）与 `reprojection_rms_px` 替代 `image`；该模式要求检测时使用同一真实内参与板参数。工具不验证外部检测来源，导出程序负责保证一致性。

## 检查和报告

工具检查有限数值、旋转矩阵、单位声明、重复样本编号、图像尺寸、静止声明、时间差、角点数量、重投影误差和旋转覆盖。随后使用 Shah 与 Li 求解，计算每组闭环残差、留一法预测误差及两种求解器差异。`ready` 要求拟合、留一法和求解器差异全部满足阈值。

内部闭环通过不代表实物绝对精度达标。最终应以未参加求解的数据及独立测量验证，再决定是否应用候选矩阵。数据接入与候选矩阵应用由现有控制程序完成。
