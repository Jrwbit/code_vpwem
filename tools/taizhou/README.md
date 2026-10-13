# 台州本轮部署工具

本轮连接 `bit-bhr@192.168.10.66` 超时，下面是已准备的软件交付和实际恢复连接后的步骤。打包成功不代表部署成功。没有真机动作、报警清除或自动上电步骤。

## 打包已提交的软件

```bash
python3 tools/taizhou/package_release.py \
  --vision /home/wjr/code/r435-rokae-migration \
  --hmi /home/wjr/code/jinlong-motor-arm-control-codex-send-stop \
  --output /home/wjr/code/code_vpwem/outputs/deployment/taizhou-20261013 \
  --evidence /tmp/taizhou-isolated-suite-20261013.log \
  --evidence /tmp/taizhou-hmi-20261013.log \
  --evidence /tmp/taizhou-native-20261013.log \
  --evidence /tmp/taizhou-ui-review-20261013.log \
  --evidence /tmp/taizhou-connectivity-20261013.log
```

工具拒绝未提交工作树和禁带资产；包是 Git HEAD 的 `.tar`、Git `.bundle`、逐源文件摘要、变更清单和检查证据。已有输出目录拒绝覆盖。`manifest.json` 的 deployment_status 固定为 prepared_only，工控机运行结果须另存真实证据。包不带 SDK 和原生二进制；本轮未改原生源码，部署前后必须核对现有二进制哈希。

## 只读核对

```bash
ssh -o BatchMode=yes -o ConnectTimeout=6 bit-bhr@192.168.10.66 \
  python3 - < tools/taizhou/probe_readonly.py
```

probe 仅读磁盘、Git、`/proc` 和已有服务 GET，不构造 SDK，不发送 POST，不采图，不读取环境凭据。输出不含 CSRF。PID 文件可能被原桌面入口放到其它位置，此时报告 unavailable，需要按当天进程和监听端口查找，不能凭历史 PID 退出服务。

核对检测 8765、HMI 50168、启动目录、Git status、实际控制器 ID、所有组件真实 off/manual/idle 和停稳、新鲜反馈、无 task/busy/capturing/latched。实时报警另由现场厂家状态核对，历史日志不是清警条件。目标工作树存在修改或不在预期提交链时，保留修改并审阅差异，不 reset、不 blanket add、不覆盖未知文件。

## 部署事务

1. 在工控机 `/home/bit-bhr/r435-release/backups/audit-20261013-<实际时间>/` 建备份目录，限制为当前用户访问。保存两个当日 HEAD、Git status 和 tracked diff；Git bundle/源码 archive；实际活动 `calibration/rokae_current.json`、`config/rokae_site.json`、相关配置；两个原生二进制；`open_real.sh`、检测 `start.sh/common.sh/open_inspection.sh`、桌面入口与 PID/运行摘要。记录这些文件的 SHA256，确认备份完整。凭据或私有配置只留在现场受限备份，不入 Git／交付源包。
2. 上传包到备份目录并 `sha256sum -c SHA256SUMS`。先读 manifest，对两个 Git bundle 执行 verify。按 probe 确认现场已空闲、安全退出且下电，不在拖动、保持、回放、采样或检测中。
3. 停止已核对的检测服务，再停止已核对的 HMI 主进程，等原生 owner、相机和监听退出。`start.sh stop` 是软件退出入口；若 PID 与实际实例不符或退出未确认，不杀其它未知进程，不继续启动第二个 owner。不能以启动第二个 SDK 连接代替结束原 owner。
4. 在两个现有仓库分别 `git fetch <对应 bundle> HEAD`，确认 FETCH_HEAD 等于 manifest revision，并与部署前 HEAD 为祖先关系；干净 tracked 工作树上 `git merge --ff-only FETCH_HEAD`。保留 ignored 活动标定／参数、现场 SDK／模型／URDF／前端资源，以及未跟踪桌面启动器；不要直接解压覆盖工作树。若 Git 拒绝覆盖本地文件，定位差异再处理，不能强制。
5. 用现场正确 Python 和依赖在新源码下跑软件检查。隔离回归可核对算法契约；`scripts/refresh_manifest.py` 重新记录真实 SDK／模型资产，然后 `verify.sh` 做实际运行包检查。缺少资产、环境导入错误或完整校验失败不算部署成功。HMI 检查使用模拟设备；不运行任何原生真机动作模式。
6. 通过实际已有 HMI `helios_hmi/run_ipc.sh --enable-write --sessions ...` 启动单一 owner，保留已有 allow-groups 限制。检测使用原 `start.sh start` 的只读默认入口；先检查 HMI 已在线再启动共享预览。不会传 `--enable-robot-motion`，不会修改现场确认状态。原入口采用 nohup 而非假定 systemd；若当天已改 systemd，使用实际查到的单一服务入口。
7. 重新 probe，核对两个磁盘 Git revision、source_sha256 与 manifest；检测 `/api/version`、HMI state.runtime_version 的 revision／摘要同样一致、dirty=false、started_at 为本次启动；PID 和实际目录一致。再次确认组件、owner、任务与运动状态，核对活动配置／标定和原生二进制哈希未被覆盖。浏览器强制重载并检查新脚本内容／无缓存。记录部署时间、日志、实际回滚位置，完成后才改交付记录为已部署。

步骤 1～7 需在真实 IPC 上执行，本轮未执行；只读 probe 不自动把任何状态评为验收通过。

## 回滚

停止并确认退出新服务，保存回滚前现场配置和历史。对两个干净 tracked 仓库分别切回**本次现场备份记录的部署前 HEAD**（例如 `git switch --detach <before-head>`），保留 ignored 现场配置；若存在新增现场源码修改，先审阅，不强制覆盖。源码、二进制和入口按本次备份哈希核对；本轮没有改二进制，通常无需替换。依据现场配置是否随后更改，审阅配置历史，不自动覆盖新参数。

重新走原珞石只读软件入口，确认运行进程版本和三部件状态；不启动旧 UR 项目。10 月 12 日旧完整包是历史应急资料，不能替代本次当日备份与真实回滚记录。
