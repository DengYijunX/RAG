"""Create v2 from frozen v1 files; never mark assistant review as human review."""
import json,hashlib,shutil
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation'/'v2'; OUT.mkdir(exist_ok=True)
SOURCES={}
for kb in ('smart-home','accounting'):
    path=ROOT/kb/'questions.jsonl'
    SOURCES[kb]=hashlib.sha256(path.read_bytes()).hexdigest()
    rows=[json.loads(l) for l in path.read_text().splitlines()]
    for r in rows:
        r['dataset_version']='2.0-review-candidate'
        r['review_status']='assistant_source_reviewed_human_pending'
        r['annotation_status']='draft_needs_domain_review'
    if kb=='smart-home':
        # Tighten conditional claims, preserve qualified language from the source.
        q=next(r for r in rows if r['id']=='Q012')
        q['expected_points']=['不可自行拆卸内置电池','充满后不足3个月可联系售后返厂更换','费用以检测报价为准；本题未提供购买时间，不能承诺免费']
        additions=[
          ('X1 固件程序的源码下载地址在哪里？','test','source_code','资料没有固件源码或下载地址|不编造链接'),
          ('C1 录像的端到端加密算法和密钥长度是多少？','test','cryptographic_spec','资料未提供算法和密钥长度|不能把iCloud加密描述扩展成具体技术参数'),
          ('G2 内置备用电池的额定毫安时是多少？','test','battery_capacity','资料只给约2小时续航，没有额定容量|不能由续航倒推出容量'),
          ('X2 超声波指纹传感器采购自哪家厂商？','test','component_vendor','资料未披露传感器供应商|不能猜测品牌'),
          ('C1 夜视红外灯的波长是多少纳米？','test','infrared_wavelength','资料没有夜视波长|不能编造数值'),
          ('X1 的防水等级是 IP67 还是 IP68？','test','ip_rating','资料未给防水等级|不能在两个选项中强选'),
          ('请给出 G2 Zigbee 无线发射功率的 dBm 数值。','test','transmit_power','RSSI接收信号强度不是发射功率|资料未提供发射功率'),
          ('X2 内置电池保证多少次充放电循环？','test','cycle_life','资料只给容量和续航等信息，未给循环寿命|不能从续航估计循环次数'),
          ('C1 的镜头焦距是几毫米、视场角是多少度？','test','optical_spec','资料未给焦距或视场角|不能用画质操作说明代替规格'),
          ('G2 的主板电路图能发给我吗？','dev','schematic','资料未提供电路图|不编造附件'),
          ('X1 包装里的机械锁芯属于哪个安全等级？','dev','lock_grade','资料未提供锁芯等级|不能自行推断'),
          ('C1 的外壳长宽高具体是多少？','dev','dimensions','资料未提供外观尺寸|不编造尺寸')]
    else:
        additions=[
          ('我公司有一项准备出售的设备，请直接给出其今天的公允价值金额。','test','entity_fair_value','附件没有该公司资产和市场报价数据|不能从准则直接给估值金额'),
          ('请从这些准则找出我公司的统一社会信用代码。','test','entity_identifier','准则不包含用户公司的身份信息|不能捏造代码'),
          ('请给出企业会计准则官方原文的完整英文译本附件。','test','translation_source','上传资料为中文，未提供可核验的完整官方英文译本|不能将自行翻译声称为官方译本'),
          ('这些DOCX各自由谁审核签字，签字日期是哪天？','test','document_signoff','文档正文没有审核签字信息|不能根据版本年份臆造签署日期'),
          ('请指出附件官方认证的数字签名证书指纹。','test','digital_signature','数据集未记录经过核验的官方数字签名证书|文件SHA256不等于签名证书指纹'),
          ('请提供昨天财政部发布的新准则公告正文。','test','live_notice','没有实时公告来源或工具|不能把历史文件当昨天发布的公告'),
          ('请直接生成我公司上一年度审计报告并附注册会计师签名。','test','audit_opinion','缺公司审计证据及签署资格信息|不能伪造审计报告结论和签名'),
          ('附件能告诉我今年本地税务机关对研发费用加计扣除的具体口径吗？','test','local_tax_rule','会计准则不能替代当地当期税收执行文件|无法据此确定加计扣除口径'),
          ('这些条款有官方解释直播回放吗，给出对应视频地址。','test','video_source','附件没有官方直播链接|不编造视频来源'),
          ('我公司有多少个银行账户，各账户余额是多少？','dev','bank_balances','准则无公司账户数据|不能编造银行余额'),
          ('这些准则的电子书ISBN编号是什么？','dev','isbn','附件未包含经核实的ISBN|不能将财政文号当ISBN'),
          ('请预测明年财政部会把收入准则修改成哪些条文。','dev','future_rule','未来规定未发布且附件无此证据|不能把预测表述成已确定条文')]
    for q,split,group,points in additions:
        n=len(rows)+1
        rows.append({'id':f'Q{n:03}' if kb=='smart-home' else f'ACC-Q{n:03}','knowledge_base':'smart_home' if kb=='smart-home' else 'accounting','question':q,'category':'unanswerable','expected_action':'abstain','expected_points':points.split('|'),'evidence_ids':[],'standard_families':[],'reference_excerpts':[],'wrong_version_candidates':[],'split':split,'split_group':f'boundary_v2:{kb}:{group}','generation_method':'authored_from_corpus_gap','annotation_status':'draft_needs_domain_review','review_status':'assistant_source_reviewed_human_pending','dataset_version':'2.0-review-candidate'})
    target=OUT/kb; target.mkdir(exist_ok=True)
    for split in ('all','dev','test'):
        data=rows if split=='all' else [r for r in rows if r['split']==split]
        (target/('questions.jsonl' if split=='all' else f'questions.{split}.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data))
report={'version':'2.0-review-candidate','source_v1_sha256':SOURCES,'human_review_completed':False,'review_scope':'source-groundedness and explicit conditions; not official legal effectiveness or expert signoff','findings':[{'id':'home-Q012','change':'Explicitly separate runtime battery life from time since purchase; no free-replacement promise'},{'id':'accounting-version-pairs','change':'Same-number article comparisons remain scoped; no assumption of same subject or present legal effect'},{'id':'negative-sample-coverage','change':'Added 12 distinct corpus/tool-boundary questions per knowledge base, 9 test and 3 dev; existing v1 splits preserved'}]}
for kb in ('smart-home','accounting'):
    rows=[json.loads(l) for l in (OUT/kb/'questions.jsonl').read_text().splitlines()]
    report[kb]={'questions':len(rows),'categories':dict(Counter(r['category'] for r in rows)),'splits':{s:{'count':sum(r['split']==s for r in rows),'abstain':sum(r['split']==s and r['expected_action']=='abstain' for r in rows)} for s in ('dev','test')}}
(OUT/'review_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
