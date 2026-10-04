import hashlib, json, re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sections = {}
files = []
for docno, path in enumerate(sorted((ROOT/'文档').glob('*.md')), 1):
    raw = path.read_bytes()
    content = raw.decode('utf-8')
    files.append({'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(raw).hexdigest()})
    matches = list(re.finditer(r'^\*\*(.+?)\*\*\s*$', content, re.M))
    for n, match in enumerate(matches, 1):
        section_id = f'D{docno}-S{n:02}'
        end = matches[n].start() if n < len(matches) else len(content)
        sections[section_id] = {'id': section_id, 'source': str(path.relative_to(ROOT)), 'heading': match.group(1), 'start_line': content[:match.start()].count('\n')+1, 'text': content[match.start():end].strip()}

rows = []
def add(question, refs, points, category='single', action='answer'):
    rows.append({'id': f'Q{len(rows)+1:03}', 'question': question, 'category': category, 'expected_action': action, 'evidence_ids': refs.split() if refs else [], 'expected_points': points.split('|'), 'annotation_status': 'draft_needs_human_review'})

# 36 single-document questions; answers are corpus-grounded, not independently verified product facts.
add('X1 换路由器后怎么重新配网，会删除指纹吗？','D1-S01','长按内面板设置键约5秒进入配对|通过智居App的WiFi重置配置2.4GHz网络|WiFi重置保留指纹和密码')
add('X1 的设置按钮完全没反应，先排查什么？','D1-S02','检查电量，低于20%更换4节新AA电池|电量正常时长按设置键约10秒强制重启|仍无响应联系售后，不自行拆修')
add('X1 一般用什么电池，快没电时怎么提醒？','D1-S04','4节AA碱性电池|约6个月续航的条件为每天开关门10次|低于20%时指示灯闪红3次并由App提醒')
add('X1 换电池后要重新录指纹吗？','D1-S05','不需要，指纹密码保留|4节同品牌全新AA电池，注意正负极')
add('X1 能连接 Google Assistant 吗？','D1-S06','不支持Google Assistant|支持小爱同学、Siri/HomeKit、天猫精灵')
add('X1 总是连不上家里的双频合一网络，怎么处理？','D1-S07','只支持2.4GHz|关闭双频合一并给2.4GHz设置独立SSID|距路由器不超过10米')
add('X1 连续短促滴滴声和间隔很长的单声分别表示什么？','D1-S09','短促连续为低电量，应换电池|长间隔单声为防撬报警，应检查撬动或撞击')
add('X1 换手机后怎样恢复控制，必须重新配对吗？','D1-S14','用新手机登录同一智居账号|无需重新配对|离线时在门锁2米内等待蓝牙重连')
add('X2 没有物理按钮，怎么重置网络？','D2-S01','屏幕底部上滑进入控制中心、设置、网络设置、WiFi重置|手机蓝牙开启，保持2米以内')
add('X2 触控屏卡死时怎样强制重启？','D2-S02','无物理设置键|长按屏幕下方Home感应区约10秒|仍无响应联系售后检测触控模块')
add('X2 充电需要多久，充电期间还能使用吗？','D2-S03','磁吸充电约3小时充满|充电时正常运行|内置5000mAh锂电池')
add('X2 充满电只能用两个月，可以自己拆换电池吗？','D2-S04','不可自行拆卸内置电池|不足3个月可联系售后返厂更换|费用以检测报价为准，不直接承诺免费')
add('X2 的 5GHz 信号只有一格，该换什么网络？','D2-S06','支持2.4GHz和5GHz|5GHz低于2格时手动选择2.4GHz|检查MAC地址过滤')
add('X2 最多能带多少个 Zigbee 传感器？','D2-S08','支持Zigbee 3.0|最多32个子设备|App的Zigbee管理中添加')
add('X2 怎么恢复出厂设置，会保留密码吗？','D2-S12','触控屏设置、系统、恢复出厂设置|清除指纹、密码和配对信息|执行前确认已备份重要设置')
add('X2 支持刷脸开锁吗？','D2-S18','不支持人脸识别|采用超声波指纹和密码开锁')
add('C1 摄像头离线，需要拔插网线吗？','D3-S01','无需拔插网线，机身没有网线接口|检查电源及2.4GHz WiFi|断电等待10秒后重新通电')
add('C1 恢复出厂设置要按哪里、多久？','D3-S02','取卡针按住底部RESET孔约5秒|听到提示音后松手|重启后需在App重新添加')
add('C1 不识别存储卡，对容量和格式有什么要求？','D3-S05','最大128GB MicroSD|FAT32格式|可在App存储管理格式化，仍失败更换已知可用卡测试')
add('C1 配对 HomeKit 时提示已经添加，应该怎么办？','D3-S07','先从家庭App删除旧设备再添加|iPhone和C1在同一2.4GHz网络|可扫码或手动输入8位配对码')
add('C1 断断续续掉线，路由器端有哪些排查项？','D3-S10','检查AP隔离|检查附近微波炉等2.4GHz干扰|尝试固定WiFi信道6或11')
add('C1 的 HomeKit 配对码丢了，还能在哪里找到？','D3-S14','智居App的C1设置、HomeKit信息|摄像头底部贴纸也有8位配对码及二维码')
add('安卓手机可以通过 HomeKit 控制 C1 吗？','D3-S17','不能，HomeKit限苹果生态|安卓使用智居App，可看实时画面、回放和双向语音')
add('第四台设备查看 C1 实时画面时提示忙，是什么限制？','D3-S18','最多3路设备同时实时预览|第4台提示设备繁忙|历史录像回放不受此限制')
add('G2 离线后应该先检查 WiFi 还是网线？','D4-S01','先检查网关与路由器的网线|检查WAN指示灯，换网线或路由器LAN口|无效再重启网关')
add('G2 恢复出厂设置后原来的子设备还在吗？','D4-S02','背面复位键约10秒，所有灯快闪3次后松手|配对子设备被清除|需要重新添加网关和全部子设备')
add('G2 在线，但所有 Zigbee 子设备一起离线，怎么排查？','D4-S04','可能是Zigbee信道严重干扰|App设置、Zigbee、信道选择中更改信道|区别于单个设备电量或距离问题')
add('G2 可以不插电长期用吗？备用电池能撑多久？','D4-S05','12V/1A适配器正常供电|备用电池约2小时|仅应急，不可长期依赖')
add('G2 升级失败一直红色快闪，有回滚办法吗？','D4-S08','升级过程中不可断电或拔网线|确认升级失败后断电30秒再通电，自动回滚|反复失败关闭自动升级并联系客服')
add('G2 Zigbee 最多几级中继，门磁可以当中继吗？','D4-S13','最多3级中继|常供电设备可作中继|电池供电门磁不参与中继')
add('G2 总是获取不同 IP，该怎么固定？','D4-S16','在路由器后台按MAC地址绑定固定IP|底部标签可找MAC|通常位于DHCP静态地址分配')
add('G2 能直接接家里的 5GHz 无线网吗？','D4-S18','不能，G2通过网线上网|可选G2-WiFi桥接配件仅支持2.4GHz|不能混同G2本体与配件能力')
add('买了五天的智居产品想无理由退货，需要哪些条件？','D5-S01','7天以内|无人为损坏且包装完好|提供订单号和购买凭证')
add('产品过保后维修有检测费吗，报价确认前会修吗？','D5-S04','检测费50元，维修成功可抵扣维修费|具体费用以售后检测报价为准|用户确认后再维修')
add('智居延保什么时候买，多少钱？','D5-S05','购买后30天内在App选购|299元每年|最多延长至三年，范围同标准保修')
add('换新与维修后的保修期分别怎么算？','D5-S10','换新从签收起重新计算一年|维修按原剩余期限，不足3个月延至3个月|延保换新后不自动转移，需联系客服')
# 8 multi-evidence cases.
add('X1 的 WiFi 重置和恢复初始状态有什么区别？','D1-S01 D1-S15','WiFi重置只改网络并保留指纹密码|恢复初始状态需设置键和开锁键同时约10秒|恢复初始状态清除指纹密码及WiFi配置','multi')
add('X1 与 X2 的重置 WiFi 操作可以照搬吗？','D1-S01 D2-S01','不能照搬|X1长按物理设置键约5秒，再用App|X2无物理设置键，走触控屏网络设置','multi')
add('C1 和 G2 都离线了，排查联网方式有什么不同？','D3-S01 D4-S01','C1检查电源及2.4GHz WiFi，无网线接口|G2检查网线、WAN灯及路由器LAN口','multi')
add('X1、X2、C1、G2 哪些可以直接用 5GHz WiFi？','D1-S07 D2-S06 D3-S03 D4-S18','X2支持5GHz|X1和C1仅2.4GHz|G2本体有线，桥接配件也只支持2.4GHz','multi')
add('X2 和 G2 的 Zigbee 子设备上限分别是多少？','D2-S08 D4-S09','X2为32个|G2为64个|不可把两个型号的上限混用','multi')
add('X1 和 X2 对 Google Assistant 的支持相同吗？','D1-S06 D2-S05','X1不支持|X2支持|两者均支持小爱同学、Siri和天猫精灵','multi')
add('买了十天的 X1 设置键故障，想直接无理由退款，该走什么流程？','D1-S02 D5-S03 D5-S06','超过7天不能按无理由退换处理|先排除电量并强制重启，仍故障联系售后|非人为故障在保修范围走维修流程|App我的、售后服务中提交故障材料','multi')
add('G2 停电三个小时再来电，需要重新添加所有传感器吗？','D4-S05 D4-S17','备用电池约2小时，3小时已超出|无需重新配置|来电后子设备可能需5到10分钟恢复','multi')
# 8 clarifications: evidence is explanatory context, not a unique answer target.
add('设备离线了，怎么办？','D3-S01 D4-S01','先问设备类型及型号|不能直接认定应拔网线或重新配WiFi','ambiguous','clarify')
add('门锁怎么重置？','D1-S01 D1-S15 D2-S01 D2-S12','先问X1还是X2，以及重置网络还是恢复出厂|涉及清除数据前说明区别','ambiguous','clarify')
add('设置键按不动，应该按多久？','D1-S02 D2-S02','先确认型号及物理按键还是触控区|不能给X2物理设置键操作','ambiguous','clarify')
add('电池坏了，能免费换吗？','D2-S10 D5-S07','需确认型号、购买时间及故障或正常衰减|信息不足不能承诺免费|若确认X2且政策冲突，交售后核实','ambiguous','clarify')
add('这个设备最多能接几个 Zigbee 传感器？','D2-S08 D4-S09','先确认X2还是G2|不能直接给统一32或64的结论','ambiguous','clarify')
add('苹果手机不能远程看摄像头了。','D3-S09 D3-S12','先确认是否C1、使用智居还是家庭App|使用家庭App时需确认家居中枢及在线状态','ambiguous','clarify')
add('门锁开始滴滴响，是不是快没电了？','D1-S09','先确认型号及蜂鸣节奏|不能把防撬报警直接判为低电量','ambiguous','clarify')
add('我想退掉产品，还能退吗？','D5-S01 D5-S03','先问购买时间、退货原因、产品与包装状态|不能无条件承诺退货','ambiguous','clarify')
# 6 unanswerable cases: no fabricated source IDs.
add('X3 门锁怎么刷脸录入？','','现有资料没有X3型号|不能套用X1或X2流程|建议提供X3说明书或咨询售后','unanswerable','abstain')
add('C1 的 RTSP 地址和默认密码是什么？','','资料未提供RTSP地址或默认密码|不能编造地址和密码','unanswerable','abstain')
add('我的订单 202610040001 现在快递到哪了？','','没有订单或物流查询工具|不能虚构配送状态|引导查看订单页或联系售后','unanswerable','abstain')
add('帮我马上远程打开家里的 X1 门锁。','','系统仅提供知识问答，没有设备控制工具|不得声称已执行开锁','unanswerable','abstain')
add('X2 最新固件具体版本号是多少？','','文档仅说明查看方法，没有实时最新版本号|不能编造版本|可引导在App或触控屏检查更新','unanswerable','abstain')
add('售后今天几点下班，联系电话是什么？','','资料没有营业时间或电话号码|不能编造|可建议从智居App售后入口查询','unanswerable','abstain')
# 2 conflicts, requiring both sources and no invented precedence rule.
add('X2 买了八个月，现在电池完全充不进电，可以免费换吗？','D2-S10 D5-S07','X2 FAQ说一年保修内无法充电等质量缺陷免费|通用政策说电池质量缺陷仅3个月内免费|明确条款冲突，资料没有优先级或生效日期|不直接承诺收费或免费，建议售后确认','conflict','flag_conflict')
add('X2 说明说 Zigbee 与 WiFi 不会互相干扰，G2 说明又说 2.4GHz WiFi 会干扰 Zigbee，该信哪个？','D2-S14 D4-S15','分别指出X2与G2文档的不同表述|不能无条件复述不会干扰|没有补充权威证据时标注资料冲突并请求核实','conflict','flag_conflict')
assert len(rows) == 60
# Split by connected components of answer evidence. No evidence section is shared
# across dev/test for answer and conflict cases. Clarifications are kept in dev.
parent = list(range(len(rows)))
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i
owners = {}
for i,r in enumerate(rows):
    if r['expected_action'] not in ('answer','flag_conflict'):
        continue
    for ref in r['evidence_ids']:
        if ref in owners:
            parent[find(i)] = find(owners[ref])
        owners[ref] = i
components = {}
for i,r in enumerate(rows):
    if r['expected_action'] != 'clarify':
        components.setdefault(find(i), []).append(i)
# Stable split, aiming for 20 holdout questions with complete evidence groups.
candidates = sorted(components.values(), key=lambda ids: hashlib.sha256(','.join(rows[i]['id'] for i in ids).encode()).hexdigest())
test = set()
for ids in candidates:
    if len(test)+len(ids) <= 20:
        test.update(ids)
for i,r in enumerate(rows):
    r['split'] = 'test' if i in test else 'dev'
    for ref in r['evidence_ids']:
        assert ref in sections, ref
(OUT/'questions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
(OUT/'evidence.jsonl').write_text(''.join(json.dumps(s,ensure_ascii=False)+'\n' for s in sections.values()))
manifest = {'schema_version':1,'repository_commit':'17943aadd3149034bf061dc0c8a5c5e2850d87f5','source_files':files,'question_count':len(rows),'section_count':len(sections),'categories':dict(Counter(r['category'] for r in rows)),'splits':dict(Counter(r['split'] for r in rows)),'annotation_status':'draft_needs_human_review','split_note':'Answer/conflict evidence components do not cross splits; clarification cases remain dev and may cite test evidence. This is a small corpus-derived development benchmark, not an untouched independent benchmark.'}
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(manifest,ensure_ascii=False,indent=2))
