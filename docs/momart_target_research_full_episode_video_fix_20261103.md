# 完整操作视频与飞书黑屏入口修正，2026-11-03

用户需要一整次操作并在完整过程中看到错误；上一版错误短片合集不满足这个要求。本次导出两条完整原始示范，各自从第0帧至最后一帧，包含选错与恢复，不拼接不同轨迹、不剪掉正确阶段。

| 完整操作 | 源帧范围/编码帧数 | 播放时长 | 错误位置 | 飞书内嵌入口 |
|---|---|---:|---|---|
| 洗碗机取碗，demo109 | 0–388 / 389帧 | 19.45秒 | 8.0–9.2秒：碗被选成碗架 | [完整操作一](https://ccnuoe6sry36.feishu.cn/docx/RfXudCiBOoeG3Bx0D34cbU1rn0g#doxcnOKH9ugrzThQmsJwzfhg3kh) |
| 洗碗机到柜子，demo42 | 0–1347 / 1348帧 | 67.4秒 | 14.4–16.0秒：碗→碗架；60.4秒拒选、60.8秒恢复 | [完整操作二](https://ccnuoe6sry36.feishu.cn/docx/RfXudCiBOoeG3Bx0D34cbU1rn0g#doxcnZfJozzAFXFTjTV4QLVEjVe) |

全部原始RGB逐帧输出，编码后逐文件计帧389/1348与源数据完全相等；源HDF actions长度也相等，最后dones均为1，但不把终止标记单独当该选择器闭环成功。播放20源step/秒，未声明已核实真实控制频率。左右面板来自同一主相机：左连续原示范，右最近一次真实查询的原图与实际选框并标held，空白查询阶段不伪造框；绿色仍是离线弱参考，红色实际预测。不是侧面/腕部，也不是模型驱动的策略闭环。

## 黑屏排查和实际修改

用户反馈飞书显示黑屏。此前只核验上传后源文件字节相同和本地ffmpeg可解码，不足以证明飞书客户端能正常播放。本轮检查旧视频的飞书720P转码，在0/2/40/126秒均有正常图像（像素均值约60–92、非全黑）；没有证据证明旧文件或旧转码本身全黑，用户端黑屏根因尚未确认。video_meta接口一次HTTP500，未将其当成视频转码成功/失败的证据。

这次采取三项具体措施：

- 使用H264 baseline/level3.1、yuv420p、avc1、无B帧、faststart重新编码；这只是实际采用的兼容性措施，不宣称已证明旧编码导致黑屏。
- 将MP4通过docs +media-insert的file-view preview直接嵌进文档播放器，不再只给云盘文件链接。回读确认两个figure的view-type均为Preview。
- 两个播放器前各插入完整时间轴GIF。第一条预览640×360/4fps/78帧，第二条480×270/3fps/202帧，保留完整操作时间轴但降低预览采样；MP4仍保留389/1348全部源帧。GIF是直接可见的备选观看方式，不是黑色视频封面或几帧截图。

## 验证范围

| 文件 | 实际播放/像素检查 |
|---|---|
| 原完整视频一 | Chrome实际video.play()成功；时间点0.1, 8.0, 19.0秒；像素均值58.0–102.1，均非黑屏；整文件ffmpeg解码成功 |
| 原完整视频二 | Chrome实际video.play()成功；时间点0.1, 14.4, 60.4, 67.0秒；像素均值61.5–107.5，均非黑屏；整文件ffmpeg解码成功 |
| 飞书视频一720P转码 | Chrome实际video.play()成功；时间点0.1, 8.0, 19.0秒；像素均值58.0–102.1，均非黑屏；整文件ffmpeg解码成功 |
| 飞书视频二360P转码 | Chrome实际video.play()成功；时间点0.1, 14.4, 60.4, 67.0秒；像素均值61.4–107.8，均非黑屏；整文件ffmpeg解码成功 |

飞书文档附件第一次token元信息查询返回981003，CLI按原token继续解析后成功列出预览；第一条720P READY并下载，第二条最初只提供360P，按实际可用360P下载验证，没有把不存在的720P当成功。两个GIF从飞书下载后SHA与本地完全一致，并核验确有78/202动画帧。已查看服务端转码错误时刻截图，红绿框和动作可见。

目前没有连接到用户已登录的飞书UI会话，因此上述证据验证服务端转码和浏览器解码，不能宣称已复现或彻底根治用户设备中的黑屏。新文档内嵌方式和完整GIF让观看不再只依赖原来的文件页播放器；最终用户设备上的表现仍需实际打开判断，不捏造客户端截图或验证。

## 复现

`build_error_videos.py --root OUTPUT_ROOT --output NEW_DIRECTORY --full-episode-ids table_setup_from_dishwasher_expert/demo_109 unload_dishwasher_to_dresser_suboptimal/demo_42`。完整模式请求episode必须全部存在且不重复，各取0..len(obs/rgb)-1；输出source_frames/rendered_frames/start/end及内容SHA。

`verify_video_playback.py --video MP4_PATH --output NEW_DIRECTORY --times .1 8 19`可验证原文件或下载后的飞书转码：整文件ffmpeg解码，隔离Chrome临时profile运行video.play并将真实解码帧绘入canvas，检查宽高、时长、playing、均值/方差，并保存实际抽帧。该工具不读取账号cookies、不冒充已登录飞书UI。

77项原相关测试通过（5条既有依赖弃用警告）；完整导出和四组原/云端浏览器媒体检查是本次直接集成验证。权重/源数据/视频/GIF/浏览器runtime输出均不入Git。

## 归档SHA

- 01_confirm7_complete_episode.mp4：49dd7f556052ea235b25bd60560ad265937212ba237f2d59e3a4f1fd22b0a63b
- 02_confirm9_complete_episode.mp4：0174c630b8f2e78857f804a7134c05ee84368d7c557420908426dd6d53c213ab
- 01_confirm7_complete_episode.gif：ee81e67edbd997f9117a354d30006333a699200831df8e798dd4bd2f4290f049
- 02_confirm9_complete_episode_small.gif：51f8947f881babcaf0129c4b2e96145d9f8e160403aa74b29397299f86450248
