# `/oss/sh2` 数据迁移：OSS 临时 STS 管理员配置指南

本文用于让管理员为 BHR → 阿里云 OSS 的数据迁移创建一套**临时、最小权限、仅限指定目录**的凭据。

目标是让用户通过 `ossutil` 直接把数据上传到 DSW 中 `/oss/sh2` 对应的 OSS 位置，从而绕过 ProxyClient/SSH 的低速链路。

> 安全要求：STS AccessKey Secret 和 SecurityToken 不要通过聊天、邮件正文或代码仓库传递。推荐由管理员直接在 BHR 终端配置；如必须交付，请使用受控的密码管理或临时密钥传递渠道。

## 一、需要管理员最终提供的信息

管理员需要确认或生成：

1. Bucket 名称
2. `/oss/sh2` 对应的 OSS 对象前缀
3. Endpoint
4. STS 临时 AccessKey ID
5. STS 临时 AccessKey Secret
6. STS SecurityToken
7. STS 到期时间（Expiration）

建议按以下格式交付：

```text
Bucket:
挂载前缀:
Endpoint:
STS AccessKey ID:
STS AccessKey Secret:
STS SecurityToken:
Expiration:
```

## 二、确认 `/oss/sh2` 对应的真实 OSS 地址

进入阿里云 PAI/DSW 控制台，找到目标 DSW 实例的存储或数据集挂载配置，查询：

```text
本地挂载点：/oss/sh2
OSS 路径：oss://<BUCKET>/<MOUNT_PREFIX>/
Region：cn-shanghai
```

例如挂载配置显示：

```text
oss://example-bucket/team/sh2/
```

则：

```text
Bucket = example-bucket
挂载前缀 = team/sh2
```

不要根据本地挂载点 `/oss/sh2` 猜测对象前缀。挂载点名称和 OSS 前缀可能不同，挂载的也可能是 Bucket 根目录。

如果 BHR 不在阿里云 VPC 内，通常使用公网 Endpoint：

```text
https://oss-cn-shanghai.aliyuncs.com
```

只有确认 BHR 能访问阿里云内网时，才使用 OSS 内网 Endpoint。

## 三、创建专用 RAM 角色

进入：

```text
RAM 访问控制
→ 身份管理
→ 角色
→ 创建角色
```

建议角色名：

```text
VPWEMBhrOssUploadRole
```

可信实体选择“阿里云账号”，并将信任策略限制为实际使用的 RAM 用户：

```json
{
  "Version": "1",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "RAM": [
          "acs:ram::<ACCOUNT_ID>:user/<RAM_USER_NAME>"
        ]
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

需要替换：

- `<ACCOUNT_ID>`：阿里云账号 ID
- `<RAM_USER_NAME>`：实际用于获取 STS 的 RAM 用户名

RAM 角色不包含永久密钥，只有信任策略允许的身份才能通过 `AssumeRole` 获取临时凭据。

## 四、创建仅限目标 OSS 前缀的权限策略

进入：

```text
RAM 访问控制
→ 权限管理
→ 权限策略
→ 创建权限策略
→ 脚本编辑
```

建议策略名：

```text
VPWEMUploadToSh2Prefix
```

如果 `/oss/sh2` 对应 `oss://<BUCKET>/<MOUNT_PREFIX>/`，使用以下策略：

```json
{
  "Version": "1",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "oss:ListObjects",
        "oss:ListMultipartUploads"
      ],
      "Resource": [
        "acs:oss:*:*:<BUCKET>"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "oss:GetObject",
        "oss:PutObject",
        "oss:AbortMultipartUpload",
        "oss:ListParts"
      ],
      "Resource": [
        "acs:oss:*:*:<BUCKET>/<MOUNT_PREFIX>/*"
      ]
    }
  ]
}
```

需要替换：

- `<BUCKET>`：实际 Bucket 名称
- `<MOUNT_PREFIX>`：`/oss/sh2` 对应的对象前缀，不带开头或结尾的 `/`

如果 `/oss/sh2` 挂载的是 Bucket 根目录，则第二段 Resource 改为：

```json
"acs:oss:*:*:<BUCKET>/*"
```

此策略允许列举、读取、上传和恢复分片上传，但没有 `oss:DeleteObject`，因此临时凭据不能删除对象。

创建策略后，将其授权给：

```text
VPWEMBhrOssUploadRole
```

## 五、允许指定 RAM 用户扮演该角色

为实际获取 STS 的 RAM 用户创建并授权以下策略：

```json
{
  "Version": "1",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "sts:AssumeRole",
      "Resource": "acs:ram::<ACCOUNT_ID>:role/VPWEMBhrOssUploadRole"
    }
  ]
}
```

建议只允许扮演这个专用角色，不要授予可扮演账号内所有角色的宽泛权限。

## 六、签发临时 STS

先将角色的最大会话时间设置为迁移所需时长。推荐设置为 12 小时，然后通过阿里云 OpenAPI Explorer 的 STS `AssumeRole` 接口，或已登录的阿里云 CLI 执行：

```bash
aliyun sts AssumeRole \
  --RoleArn acs:ram::<ACCOUNT_ID>:role/VPWEMBhrOssUploadRole \
  --RoleSessionName vpwem-bhr-migration \
  --DurationSeconds 43200
```

返回值中需要记录：

```text
Credentials.AccessKeyId
Credentials.AccessKeySecret
Credentials.SecurityToken
Credentials.Expiration
```

如果角色最大会话时间低于 12 小时，请将 `DurationSeconds` 调整到允许范围内。STS 到期后可以重新签发；重新配置凭据后，`ossutil` 可以继续上传并跳过已完成的内容。

## 七、推荐的交付方式

最安全的方式是管理员直接登录 BHR，在用户指定的受控位置配置 `ossutil`，避免把 STS Secret 和 Token 发给用户。

如果必须交付凭据：

- Bucket、前缀和 Endpoint 可以正常发送。
- AccessKey Secret 和 SecurityToken 应通过安全渠道发送。
- 必须同时告知 Expiration。
- 不要把凭据写进脚本、Git 仓库、实验配置或训练日志。

## 八、迁移前验证

管理员或用户配置好 `ossutil` 后，应先做小文件验证：

1. 能列出指定前缀。
2. 能向指定前缀上传一个小测试文件。
3. 不能访问无关前缀。
4. 不能删除对象。

确认直传可用后，再开始大规模迁移。开始 OSS 直传前，应先停止当前经过 ProxyClient/SSH 的慢速传输，避免两条链路同时写入同名大文件。

## 官方参考资料

- [RAM 角色概述](https://www.alibabacloud.com/help/en/ram/user-guide/ram-role-overview)
- [创建 RAM 角色并授权](https://www.alibabacloud.com/help/en/ram/create-a-ram-role-and-authorize-it)
- [扮演 RAM 角色](https://www.alibabacloud.com/help/en/ram/user-guide/assume-a-ram-role)
- [STS 临时访问授权](https://www.alibabacloud.com/help/en/oss/sts-temporary-access-authorization)
- [OSS 分片上传及所需权限](https://www.alibabacloud.com/help/en/oss/user-guide/multipart-upload)
- [ossutil 配置说明](https://www.alibabacloud.com/help/en/oss/developer-reference/config)
