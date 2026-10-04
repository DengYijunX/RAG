> 已解决：最新发布记录的密钥目标与阿里云域名一致，Embedding及LLM认证已通过。以下为历史排查记录。

# Embedding 配置修正

两个字段用途不同，需分别保存：

- 环境变量 `EMBEDDING_BASE_URL`：
  `https://llm-k5ve8qaaz1tn5w7b.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`
- 现有密钥 `EMBEDDING_API_KEY` 的允许目标域名：
  `llm-k5ve8qaaz1tn5w7b.cn-beijing.maas.aliyuncs.com`

最新检查：当前URL配置不是完整HTTPS地址；临时覆盖成上述完整URL后返回401。密钥绑定存在，但目标仍为api.siliconflow.cn。配置工具禁止改写既有密钥目标，需在环境设置编辑该项后保存并发布。无需发送密钥到聊天或创建重复密钥。

模型变量已存在，仍需认证成功后确认该部署接受该模型。LLM最小请求在此前环境检查中已通过；真实检索基线未运行。
