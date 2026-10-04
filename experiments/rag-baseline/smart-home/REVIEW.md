# 智能家居售后评测题审阅稿

状态：基于资料编写的草稿，待领域人工复核。dev/test 为分组隔离后的初始划分。不得将本文件或 questions*.jsonl 索引进知识库。

## Q001 · single · test

X1 换路由器后怎么重新配网，会删除指纹吗？

预期行为：`answer`

- 长按内面板设置键约5秒进入配对
- 通过智居App的WiFi重置配置2.4GHz网络
- WiFi重置保留指纹和密码

- 证据 `D1-S01`：[ Doc1-X1智能门锁FAQ(1).md · 怎么重置 WiFi？ ](<corpus/Doc1-X1智能门锁FAQ(1).md>)

## Q002 · single · dev

X1 的设置按钮完全没反应，先排查什么？

预期行为：`answer`

- 检查电量，低于20%更换4节新AA电池
- 电量正常时长按设置键约10秒强制重启
- 仍无响应联系售后，不自行拆修

- 证据 `D1-S02`：[ Doc1-X1智能门锁FAQ(1).md · 设置键按了没反应怎么办？ ](<corpus/Doc1-X1智能门锁FAQ(1).md>)

## Q003 · single · test

X1 一般用什么电池，快没电时怎么提醒？

预期行为：`answer`

- 4节AA碱性电池
- 约6个月续航的条件为每天开关门10次
- 低于20%时指示灯闪红3次并由App提醒

- 证据 `D1-S04`：[ Doc1-X1智能门锁FAQ(1).md · 电池能用多久？ ](<corpus/Doc1-X1智能门锁FAQ(1).md>)

## Q004 · single · test

X1 换电池后要重新录指纹吗？

预期行为：`answer`

- 不需要，指纹密码保留
- 4节同品牌全新AA电池，注意正负极

- 证据 `D1-S05`：[ Doc1-X1智能门锁FAQ(1).md · 怎么更换电池？ ](<corpus/Doc1-X1智能门锁FAQ(1).md>)

## Q005 · single · test

X1 能连接 Google Assistant 吗？

预期行为：`answer`

- 不支持Google Assistant
- 支持小爱同学、Siri/HomeKit、天猫精灵

- 证据 `D1-S06`：[ Doc1-X1智能门锁FAQ(1).md · 支持哪些语音助手？ ](<corpus/Doc1-X1智能门锁FAQ(1).md>)

## Q006 · single · dev

X1 总是连不上家里的双频合一网络，怎么处理？

预期行为：`answer`

- 只支持2.4GHz
- 关闭双频合一并给2.4GHz设置独立SSID
- 距路由器不超过10米

- 证据 `D1-S07`：[ Doc1-X1智能门锁FAQ(1).md · 连不上 WiFi 怎么办？ ](<corpus/Doc1-X1智能门锁FAQ(1).md>)

## Q007 · single · dev

X1 连续短促滴滴声和间隔很长的单声分别表示什么？

预期行为：`answer`

- 短促连续为低电量，应换电池
- 长间隔单声为防撬报警，应检查撬动或撞击

- 证据 `D1-S09`：[ Doc1-X1智能门锁FAQ(1).md · 门锁一直滴滴响？ ](<corpus/Doc1-X1智能门锁FAQ(1).md>)

## Q008 · single · dev

X1 换手机后怎样恢复控制，必须重新配对吗？

预期行为：`answer`

- 用新手机登录同一智居账号
- 无需重新配对
- 离线时在门锁2米内等待蓝牙重连

- 证据 `D1-S14`：[ Doc1-X1智能门锁FAQ(1).md · 换了手机 App 怎么重新绑定门锁？ ](<corpus/Doc1-X1智能门锁FAQ(1).md>)

## Q009 · single · test

X2 没有物理按钮，怎么重置网络？

预期行为：`answer`

- 屏幕底部上滑进入控制中心、设置、网络设置、WiFi重置
- 手机蓝牙开启，保持2米以内

- 证据 `D2-S01`：[ Doc2-X2智能门锁FAQ.md · 怎么重置 WiFi？ ](<corpus/Doc2-X2智能门锁FAQ.md>)

## Q010 · single · dev

X2 触控屏卡死时怎样强制重启？

预期行为：`answer`

- 无物理设置键
- 长按屏幕下方Home感应区约10秒
- 仍无响应联系售后检测触控模块

- 证据 `D2-S02`：[ Doc2-X2智能门锁FAQ.md · 设置键按了没反应怎么办？ ](<corpus/Doc2-X2智能门锁FAQ.md>)

## Q011 · single · dev

X2 充电需要多久，充电期间还能使用吗？

预期行为：`answer`

- 磁吸充电约3小时充满
- 充电时正常运行
- 内置5000mAh锂电池

- 证据 `D2-S03`：[ Doc2-X2智能门锁FAQ.md · 电池能用多久？ ](<corpus/Doc2-X2智能门锁FAQ.md>)

## Q012 · single · test

X2 充满电只能用两个月，可以自己拆换电池吗？

预期行为：`answer`

- 不可自行拆卸内置电池
- 不足3个月可联系售后返厂更换
- 费用以检测报价为准，不直接承诺免费

- 证据 `D2-S04`：[ Doc2-X2智能门锁FAQ.md · 怎么更换电池？ ](<corpus/Doc2-X2智能门锁FAQ.md>)

## Q013 · single · dev

X2 的 5GHz 信号只有一格，该换什么网络？

预期行为：`answer`

- 支持2.4GHz和5GHz
- 5GHz低于2格时手动选择2.4GHz
- 检查MAC地址过滤

- 证据 `D2-S06`：[ Doc2-X2智能门锁FAQ.md · 连不上 WiFi 怎么办？ ](<corpus/Doc2-X2智能门锁FAQ.md>)

## Q014 · single · dev

X2 最多能带多少个 Zigbee 传感器？

预期行为：`answer`

- 支持Zigbee 3.0
- 最多32个子设备
- App的Zigbee管理中添加

- 证据 `D2-S08`：[ Doc2-X2智能门锁FAQ.md · 支持 Zigbee 吗？ ](<corpus/Doc2-X2智能门锁FAQ.md>)

## Q015 · single · test

X2 怎么恢复出厂设置，会保留密码吗？

预期行为：`answer`

- 触控屏设置、系统、恢复出厂设置
- 清除指纹、密码和配对信息
- 执行前确认已备份重要设置

- 证据 `D2-S12`：[ Doc2-X2智能门锁FAQ.md · X2 怎么恢复出厂设置？ ](<corpus/Doc2-X2智能门锁FAQ.md>)

## Q016 · single · dev

X2 支持刷脸开锁吗？

预期行为：`answer`

- 不支持人脸识别
- 采用超声波指纹和密码开锁

- 证据 `D2-S18`：[ Doc2-X2智能门锁FAQ.md · X2 支持人脸识别吗？ ](<corpus/Doc2-X2智能门锁FAQ.md>)

## Q017 · single · dev

C1 摄像头离线，需要拔插网线吗？

预期行为：`answer`

- 无需拔插网线，机身没有网线接口
- 检查电源及2.4GHz WiFi
- 断电等待10秒后重新通电

- 证据 `D3-S01`：[ Doc3-C1智能摄像头FAQ.md · 设备离线怎么办？ ](<corpus/Doc3-C1智能摄像头FAQ.md>)

## Q018 · single · dev

C1 恢复出厂设置要按哪里、多久？

预期行为：`answer`

- 取卡针按住底部RESET孔约5秒
- 听到提示音后松手
- 重启后需在App重新添加

- 证据 `D3-S02`：[ Doc3-C1智能摄像头FAQ.md · 怎么重置？ ](<corpus/Doc3-C1智能摄像头FAQ.md>)

## Q019 · single · test

C1 不识别存储卡，对容量和格式有什么要求？

预期行为：`answer`

- 最大128GB MicroSD
- FAT32格式
- 可在App存储管理格式化，仍失败更换已知可用卡测试

- 证据 `D3-S05`：[ Doc3-C1智能摄像头FAQ.md · 存储卡不识别？ ](<corpus/Doc3-C1智能摄像头FAQ.md>)

## Q020 · single · test

C1 配对 HomeKit 时提示已经添加，应该怎么办？

预期行为：`answer`

- 先从家庭App删除旧设备再添加
- iPhone和C1在同一2.4GHz网络
- 可扫码或手动输入8位配对码

- 证据 `D3-S07`：[ Doc3-C1智能摄像头FAQ.md · iPhone 连不上 HomeKit 怎么办？ ](<corpus/Doc3-C1智能摄像头FAQ.md>)

## Q021 · single · dev

C1 断断续续掉线，路由器端有哪些排查项？

预期行为：`answer`

- 检查AP隔离
- 检查附近微波炉等2.4GHz干扰
- 尝试固定WiFi信道6或11

- 证据 `D3-S10`：[ Doc3-C1智能摄像头FAQ.md · 摄像头频繁掉线怎么解决？ ](<corpus/Doc3-C1智能摄像头FAQ.md>)

## Q022 · single · dev

C1 的 HomeKit 配对码丢了，还能在哪里找到？

预期行为：`answer`

- 智居App的C1设置、HomeKit信息
- 摄像头底部贴纸也有8位配对码及二维码

- 证据 `D3-S14`：[ Doc3-C1智能摄像头FAQ.md · HomeKit 配对码丢失了怎么办？ ](<corpus/Doc3-C1智能摄像头FAQ.md>)

## Q023 · single · dev

安卓手机可以通过 HomeKit 控制 C1 吗？

预期行为：`answer`

- 不能，HomeKit限苹果生态
- 安卓使用智居App，可看实时画面、回放和双向语音

- 证据 `D3-S17`：[ Doc3-C1智能摄像头FAQ.md · 安卓手机能用 HomeKit 功能吗？ ](<corpus/Doc3-C1智能摄像头FAQ.md>)

## Q024 · single · dev

第四台设备查看 C1 实时画面时提示忙，是什么限制？

预期行为：`answer`

- 最多3路设备同时实时预览
- 第4台提示设备繁忙
- 历史录像回放不受此限制

- 证据 `D3-S18`：[ Doc3-C1智能摄像头FAQ.md · 多个苹果设备同时查看摄像头会卡吗？ ](<corpus/Doc3-C1智能摄像头FAQ.md>)

## Q025 · single · dev

G2 离线后应该先检查 WiFi 还是网线？

预期行为：`answer`

- 先检查网关与路由器的网线
- 检查WAN指示灯，换网线或路由器LAN口
- 无效再重启网关

- 证据 `D4-S01`：[ Doc4-G2智能网关FAQ(1).md · 设备离线怎么办？ ](<corpus/Doc4-G2智能网关FAQ(1).md>)

## Q026 · single · dev

G2 恢复出厂设置后原来的子设备还在吗？

预期行为：`answer`

- 背面复位键约10秒，所有灯快闪3次后松手
- 配对子设备被清除
- 需要重新添加网关和全部子设备

- 证据 `D4-S02`：[ Doc4-G2智能网关FAQ(1).md · 怎么重置？ ](<corpus/Doc4-G2智能网关FAQ(1).md>)

## Q027 · single · dev

G2 在线，但所有 Zigbee 子设备一起离线，怎么排查？

预期行为：`answer`

- 可能是Zigbee信道严重干扰
- App设置、Zigbee、信道选择中更改信道
- 区别于单个设备电量或距离问题

- 证据 `D4-S04`：[ Doc4-G2智能网关FAQ(1).md · 子设备全部离线但网关在线？ ](<corpus/Doc4-G2智能网关FAQ(1).md>)

## Q028 · single · dev

G2 可以不插电长期用吗？备用电池能撑多久？

预期行为：`answer`

- 12V/1A适配器正常供电
- 备用电池约2小时
- 仅应急，不可长期依赖

- 证据 `D4-S05`：[ Doc4-G2智能网关FAQ(1).md · 电源和备用电池？ ](<corpus/Doc4-G2智能网关FAQ(1).md>)

## Q029 · single · dev

G2 升级失败一直红色快闪，有回滚办法吗？

预期行为：`answer`

- 升级过程中不可断电或拔网线
- 确认升级失败后断电30秒再通电，自动回滚
- 反复失败关闭自动升级并联系客服

- 证据 `D4-S08`：[ Doc4-G2智能网关FAQ(1).md · 网关固件升级失败怎么办？ ](<corpus/Doc4-G2智能网关FAQ(1).md>)

## Q030 · single · dev

G2 Zigbee 最多几级中继，门磁可以当中继吗？

预期行为：`answer`

- 最多3级中继
- 常供电设备可作中继
- 电池供电门磁不参与中继

- 证据 `D4-S13`：[ Doc4-G2智能网关FAQ(1).md · Zigbee 网络支持几级中继？ ](<corpus/Doc4-G2智能网关FAQ(1).md>)

## Q031 · single · test

G2 总是获取不同 IP，该怎么固定？

预期行为：`answer`

- 在路由器后台按MAC地址绑定固定IP
- 底部标签可找MAC
- 通常位于DHCP静态地址分配

- 证据 `D4-S16`：[ Doc4-G2智能网关FAQ(1).md · 网关 IP 地址总是变怎么办？ ](<corpus/Doc4-G2智能网关FAQ(1).md>)

## Q032 · single · dev

G2 能直接接家里的 5GHz 无线网吗？

预期行为：`answer`

- 不能，G2通过网线上网
- 可选G2-WiFi桥接配件仅支持2.4GHz
- 不能混同G2本体与配件能力

- 证据 `D4-S18`：[ Doc4-G2智能网关FAQ(1).md · 网关能连接 5GHz WiFi 吗？ ](<corpus/Doc4-G2智能网关FAQ(1).md>)

## Q033 · single · dev

买了五天的智居产品想无理由退货，需要哪些条件？

预期行为：`answer`

- 7天以内
- 无人为损坏且包装完好
- 提供订单号和购买凭证

- 证据 `D5-S01`：[ Doc5-售后与保修政策.md · 退换货政策 ](<corpus/Doc5-售后与保修政策.md>)

## Q034 · single · dev

产品过保后维修有检测费吗，报价确认前会修吗？

预期行为：`answer`

- 检测费50元，维修成功可抵扣维修费
- 具体费用以售后检测报价为准
- 用户确认后再维修

- 证据 `D5-S04`：[ Doc5-售后与保修政策.md · 过保后维修怎么收费？ ](<corpus/Doc5-售后与保修政策.md>)

## Q035 · single · test

智居延保什么时候买，多少钱？

预期行为：`answer`

- 购买后30天内在App选购
- 299元每年
- 最多延长至三年，范围同标准保修

- 证据 `D5-S05`：[ Doc5-售后与保修政策.md · 延保服务 ](<corpus/Doc5-售后与保修政策.md>)

## Q036 · single · dev

换新与维修后的保修期分别怎么算？

预期行为：`answer`

- 换新从签收起重新计算一年
- 维修按原剩余期限，不足3个月延至3个月
- 延保换新后不自动转移，需联系客服

- 证据 `D5-S10`：[ Doc5-售后与保修政策.md · 换新后的产品保修期怎么计算？ ](<corpus/Doc5-售后与保修政策.md>)

## Q037 · multi · test

X1 的 WiFi 重置和恢复初始状态有什么区别？

预期行为：`answer`

- WiFi重置只改网络并保留指纹密码
- 恢复初始状态需设置键和开锁键同时约10秒
- 恢复初始状态清除指纹密码及WiFi配置

- 证据 `D1-S01`：[ Doc1-X1智能门锁FAQ(1).md · 怎么重置 WiFi？ ](<corpus/Doc1-X1智能门锁FAQ(1).md>)
- 证据 `D1-S15`：[ Doc1-X1智能门锁FAQ(1).md · 怎么把门锁恢复到初始状态？ ](<corpus/Doc1-X1智能门锁FAQ(1).md>)

## Q038 · multi · test

X1 与 X2 的重置 WiFi 操作可以照搬吗？

预期行为：`answer`

- 不能照搬
- X1长按物理设置键约5秒，再用App
- X2无物理设置键，走触控屏网络设置

- 证据 `D1-S01`：[ Doc1-X1智能门锁FAQ(1).md · 怎么重置 WiFi？ ](<corpus/Doc1-X1智能门锁FAQ(1).md>)
- 证据 `D2-S01`：[ Doc2-X2智能门锁FAQ.md · 怎么重置 WiFi？ ](<corpus/Doc2-X2智能门锁FAQ.md>)

## Q039 · multi · dev

C1 和 G2 都离线了，排查联网方式有什么不同？

预期行为：`answer`

- C1检查电源及2.4GHz WiFi，无网线接口
- G2检查网线、WAN灯及路由器LAN口

- 证据 `D3-S01`：[ Doc3-C1智能摄像头FAQ.md · 设备离线怎么办？ ](<corpus/Doc3-C1智能摄像头FAQ.md>)
- 证据 `D4-S01`：[ Doc4-G2智能网关FAQ(1).md · 设备离线怎么办？ ](<corpus/Doc4-G2智能网关FAQ(1).md>)

## Q040 · multi · dev

X1、X2、C1、G2 哪些可以直接用 5GHz WiFi？

预期行为：`answer`

- X2支持5GHz
- X1和C1仅2.4GHz
- G2本体有线，桥接配件也只支持2.4GHz

- 证据 `D1-S07`：[ Doc1-X1智能门锁FAQ(1).md · 连不上 WiFi 怎么办？ ](<corpus/Doc1-X1智能门锁FAQ(1).md>)
- 证据 `D2-S06`：[ Doc2-X2智能门锁FAQ.md · 连不上 WiFi 怎么办？ ](<corpus/Doc2-X2智能门锁FAQ.md>)
- 证据 `D3-S03`：[ Doc3-C1智能摄像头FAQ.md · 连不上 WiFi 怎么办？ ](<corpus/Doc3-C1智能摄像头FAQ.md>)
- 证据 `D4-S18`：[ Doc4-G2智能网关FAQ(1).md · 网关能连接 5GHz WiFi 吗？ ](<corpus/Doc4-G2智能网关FAQ(1).md>)

## Q041 · multi · dev

X2 和 G2 的 Zigbee 子设备上限分别是多少？

预期行为：`answer`

- X2为32个
- G2为64个
- 不可把两个型号的上限混用

- 证据 `D2-S08`：[ Doc2-X2智能门锁FAQ.md · 支持 Zigbee 吗？ ](<corpus/Doc2-X2智能门锁FAQ.md>)
- 证据 `D4-S09`：[ Doc4-G2智能网关FAQ(1).md · 支持多少子设备？ ](<corpus/Doc4-G2智能网关FAQ(1).md>)

## Q042 · multi · test

X1 和 X2 对 Google Assistant 的支持相同吗？

预期行为：`answer`

- X1不支持
- X2支持
- 两者均支持小爱同学、Siri和天猫精灵

- 证据 `D1-S06`：[ Doc1-X1智能门锁FAQ(1).md · 支持哪些语音助手？ ](<corpus/Doc1-X1智能门锁FAQ(1).md>)
- 证据 `D2-S05`：[ Doc2-X2智能门锁FAQ.md · 支持哪些语音助手？ ](<corpus/Doc2-X2智能门锁FAQ.md>)

## Q043 · multi · dev

买了十天的 X1 设置键故障，想直接无理由退款，该走什么流程？

预期行为：`answer`

- 超过7天不能按无理由退换处理
- 先排除电量并强制重启，仍故障联系售后
- 非人为故障在保修范围走维修流程
- App我的、售后服务中提交故障材料

- 证据 `D1-S02`：[ Doc1-X1智能门锁FAQ(1).md · 设置键按了没反应怎么办？ ](<corpus/Doc1-X1智能门锁FAQ(1).md>)
- 证据 `D5-S03`：[ Doc5-售后与保修政策.md · 维修和退换有什么区别？ ](<corpus/Doc5-售后与保修政策.md>)
- 证据 `D5-S06`：[ Doc5-售后与保修政策.md · 售后申请流程 ](<corpus/Doc5-售后与保修政策.md>)

## Q044 · multi · dev

G2 停电三个小时再来电，需要重新添加所有传感器吗？

预期行为：`answer`

- 备用电池约2小时，3小时已超出
- 无需重新配置
- 来电后子设备可能需5到10分钟恢复

- 证据 `D4-S05`：[ Doc4-G2智能网关FAQ(1).md · 电源和备用电池？ ](<corpus/Doc4-G2智能网关FAQ(1).md>)
- 证据 `D4-S17`：[ Doc4-G2智能网关FAQ(1).md · 停电后网关需要重新配置吗？ ](<corpus/Doc4-G2智能网关FAQ(1).md>)

## Q045 · ambiguous · dev

设备离线了，怎么办？

预期行为：`clarify`

- 先问设备类型及型号
- 不能直接认定应拔网线或重新配WiFi

- 证据 `D3-S01`：[ Doc3-C1智能摄像头FAQ.md · 设备离线怎么办？ ](<corpus/Doc3-C1智能摄像头FAQ.md>)
- 证据 `D4-S01`：[ Doc4-G2智能网关FAQ(1).md · 设备离线怎么办？ ](<corpus/Doc4-G2智能网关FAQ(1).md>)

## Q046 · ambiguous · test

门锁怎么重置？

预期行为：`clarify`

- 先问X1还是X2，以及重置网络还是恢复出厂
- 涉及清除数据前说明区别

- 证据 `D1-S01`：[ Doc1-X1智能门锁FAQ(1).md · 怎么重置 WiFi？ ](<corpus/Doc1-X1智能门锁FAQ(1).md>)
- 证据 `D1-S15`：[ Doc1-X1智能门锁FAQ(1).md · 怎么把门锁恢复到初始状态？ ](<corpus/Doc1-X1智能门锁FAQ(1).md>)
- 证据 `D2-S01`：[ Doc2-X2智能门锁FAQ.md · 怎么重置 WiFi？ ](<corpus/Doc2-X2智能门锁FAQ.md>)
- 证据 `D2-S12`：[ Doc2-X2智能门锁FAQ.md · X2 怎么恢复出厂设置？ ](<corpus/Doc2-X2智能门锁FAQ.md>)

## Q047 · ambiguous · dev

设置键按不动，应该按多久？

预期行为：`clarify`

- 先确认型号及物理按键还是触控区
- 不能给X2物理设置键操作

- 证据 `D1-S02`：[ Doc1-X1智能门锁FAQ(1).md · 设置键按了没反应怎么办？ ](<corpus/Doc1-X1智能门锁FAQ(1).md>)
- 证据 `D2-S02`：[ Doc2-X2智能门锁FAQ.md · 设置键按了没反应怎么办？ ](<corpus/Doc2-X2智能门锁FAQ.md>)

## Q048 · ambiguous · test

电池坏了，能免费换吗？

预期行为：`clarify`

- 需确认型号、购买时间及故障或正常衰减
- 信息不足不能承诺免费
- 若确认X2且政策冲突，交售后核实

- 证据 `D2-S10`：[ Doc2-X2智能门锁FAQ.md · 保修期内换电池免费吗？ ](<corpus/Doc2-X2智能门锁FAQ.md>)
- 证据 `D5-S07`：[ Doc5-售后与保修政策.md · 电池属于保修范围吗？ ](<corpus/Doc5-售后与保修政策.md>)

## Q049 · ambiguous · dev

这个设备最多能接几个 Zigbee 传感器？

预期行为：`clarify`

- 先确认X2还是G2
- 不能直接给统一32或64的结论

- 证据 `D2-S08`：[ Doc2-X2智能门锁FAQ.md · 支持 Zigbee 吗？ ](<corpus/Doc2-X2智能门锁FAQ.md>)
- 证据 `D4-S09`：[ Doc4-G2智能网关FAQ(1).md · 支持多少子设备？ ](<corpus/Doc4-G2智能网关FAQ(1).md>)

## Q050 · ambiguous · test

苹果手机不能远程看摄像头了。

预期行为：`clarify`

- 先确认是否C1、使用智居还是家庭App
- 使用家庭App时需确认家居中枢及在线状态

- 证据 `D3-S09`：[ Doc3-C1智能摄像头FAQ.md · 怎么在苹果手机上远程查看摄像头？ ](<corpus/Doc3-C1智能摄像头FAQ.md>)
- 证据 `D3-S12`：[ Doc3-C1智能摄像头FAQ.md · Apple TV 作 HomeKit 中枢后无法远程查看？ ](<corpus/Doc3-C1智能摄像头FAQ.md>)

## Q051 · ambiguous · dev

门锁开始滴滴响，是不是快没电了？

预期行为：`clarify`

- 先确认型号及蜂鸣节奏
- 不能把防撬报警直接判为低电量

- 证据 `D1-S09`：[ Doc1-X1智能门锁FAQ(1).md · 门锁一直滴滴响？ ](<corpus/Doc1-X1智能门锁FAQ(1).md>)

## Q052 · ambiguous · dev

我想退掉产品，还能退吗？

预期行为：`clarify`

- 先问购买时间、退货原因、产品与包装状态
- 不能无条件承诺退货

- 证据 `D5-S01`：[ Doc5-售后与保修政策.md · 退换货政策 ](<corpus/Doc5-售后与保修政策.md>)
- 证据 `D5-S03`：[ Doc5-售后与保修政策.md · 维修和退换有什么区别？ ](<corpus/Doc5-售后与保修政策.md>)

## Q053 · unanswerable · dev

X3 门锁怎么刷脸录入？

预期行为：`abstain`

- 现有资料没有X3型号
- 不能套用X1或X2流程
- 建议提供X3说明书或咨询售后


## Q054 · unanswerable · test

C1 的 RTSP 地址和默认密码是什么？

预期行为：`abstain`

- 资料未提供RTSP地址或默认密码
- 不能编造地址和密码


## Q055 · unanswerable · dev

我的订单 202610040001 现在快递到哪了？

预期行为：`abstain`

- 没有订单或物流查询工具
- 不能虚构配送状态
- 引导查看订单页或联系售后


## Q056 · unanswerable · dev

帮我马上远程打开家里的 X1 门锁。

预期行为：`abstain`

- 系统仅提供知识问答，没有设备控制工具
- 不得声称已执行开锁


## Q057 · unanswerable · dev

X2 最新固件具体版本号是多少？

预期行为：`abstain`

- 文档仅说明查看方法，没有实时最新版本号
- 不能编造版本
- 可引导在App或触控屏检查更新


## Q058 · unanswerable · dev

售后今天几点下班，联系电话是什么？

预期行为：`abstain`

- 资料没有营业时间或电话号码
- 不能编造
- 可建议从智居App售后入口查询


## Q059 · conflict · test

X2 买了八个月，现在电池完全充不进电，可以免费换吗？

预期行为：`flag_conflict`

- X2 FAQ说一年保修内无法充电等质量缺陷免费
- 通用政策说电池质量缺陷仅3个月内免费
- 明确条款冲突，资料没有优先级或生效日期
- 不直接承诺收费或免费，建议售后确认

- 证据 `D2-S10`：[ Doc2-X2智能门锁FAQ.md · 保修期内换电池免费吗？ ](<corpus/Doc2-X2智能门锁FAQ.md>)
- 证据 `D5-S07`：[ Doc5-售后与保修政策.md · 电池属于保修范围吗？ ](<corpus/Doc5-售后与保修政策.md>)

## Q060 · conflict · test

X2 说明说 Zigbee 与 WiFi 不会互相干扰，G2 说明又说 2.4GHz WiFi 会干扰 Zigbee，该信哪个？

预期行为：`flag_conflict`

- 分别指出X2与G2文档的不同表述
- 不能无条件复述不会干扰
- 没有补充权威证据时标注资料冲突并请求核实

- 证据 `D2-S14`：[ Doc2-X2智能门锁FAQ.md · X2 的 Zigbee 和 WiFi 有什么区别？ ](<corpus/Doc2-X2智能门锁FAQ.md>)
- 证据 `D4-S15`：[ Doc4-G2智能网关FAQ(1).md · 如何检测 Zigbee 信号干扰？ ](<corpus/Doc4-G2智能网关FAQ(1).md>)
