import json, hashlib, re
from pathlib import Path
from collections import Counter, defaultdict
OUT=Path(__file__).resolve().parents[1]
def read(name): return [json.loads(l) for l in (OUT/name).read_text().splitlines()]
articles=read('articles.jsonl'); docs=read('documents.jsonl')
by_id={a['id']:a for a in articles}; by_doc=defaultdict(list)
for a in articles: by_doc[a['document_id']].append(a)
rows=[]
def add(q,refs,points,category='semantic',action='answer',families=None,origin='authored_from_source'):
    refs=refs.split() if isinstance(refs,str) else refs
    evidence=[by_id[k] for k in refs]
    groups=sorted(set(families or [a['standard_id'] for a in evidence]))
    negatives=[a['id'] for a in articles if any(a['standard_id']==e['standard_id'] and a['version_year']!=e['version_year'] and a['article_number']==e['article_number'] for e in evidence) and a['id'] not in refs]
    rows.append({'id':f'ACC-Q{len(rows)+1:03}','knowledge_base':'accounting','question':q,'category':category,'expected_action':action,'expected_points':points.split('|') if isinstance(points,str) else points,'evidence_ids':refs,'reference_excerpts':[{'article_id':a['id'],'text':a['text']} for a in evidence],'wrong_version_candidates':negatives,'standard_families':groups,'generation_method':origin,'annotation_status':'draft_needs_domain_review','authority_scope':'uploaded documents only; current legal effectiveness not verified'})
# Coverage probes: explicitly easy, not counted as semantic reasoning tests.
for d in docs:
    choices=[a for a in by_doc[d['id']] if 2<=a['article_number']<=10 and len(a['text'])<1600]
    a=next((a for a in choices if '是指' in a['text']),choices[0])
    add(f'请查找《{d["title"]}》{a["article_label"]}，概括其规定并引用该年份版本的原文。',[a['id']],['准确定位指定准则、年份和条号','概括所附原文，不用其他年份替代'],category='article_lookup',origin='template_from_article_metadata')
# Authored semantic and numerical scenarios. All versions explicitly scoped.
add('按2006年存货准则，已经入库的货物只要数量可查就能确认存货吗？','CAS-01-2006-A004','不能仅凭数量|经济利益很可能流入|成本能够可靠计量')
add('按2006年存货准则，异常浪费的材料费和普通仓储费能加进存货成本吗？','CAS-01-2006-A009','非正常消耗费用计入当期损益|普通仓储费用不计入成本|为下一生产阶段必需的仓储费用例外')
add('按2006年存货准则，成本100万元，预计售价95万元，完工成本3万元、销售费用和税费2万元，期末如何计量？','CAS-01-2006-A015','可变现净值为95-3-2=90万元|成本与可变现净值孰低，期末90万元|计提跌价准备10万元','calculation')
add('按2006年存货准则，原来提了10万元跌价准备，影响因素消失后最多能转回多少？','CAS-01-2006-A019','转回以原计提10万元为上限|实际转回额还取决于恢复情况|计入当期损益','calculation')
add('按2006年存货准则，为特定项目专门购入、不可替代的货物通常用什么发出计价方法？','CAS-01-2006-A014','通常采用个别计价法|不能把通用计价方法不加条件套用')
add('按2006年固定资产准则，已提足折旧但仍在使用的设备还要继续折旧吗？','CAS-04-2006-A014','不再计提折旧|单独计价入账土地也是例外')
add('按2006年固定资产准则，设备成本120万元，预计净残值10万元，累计减值准备20万元，应计折旧额是多少？','CAS-04-2006-A014','应计折旧额120-10-20=90万元|不是本期折旧费用，缺年限和方法不能推算本期折旧','calculation')
add('按2006年固定资产准则，同一设备的不同部件寿命差异很大，必须统一折旧吗？','CAS-04-2006-A005','符合不同使用寿命或利益实现方式、适用不同折旧率或方法时|分别确认为单项固定资产')
add('按2006年无形资产准则，研究阶段花的钱能因为项目将来成功而先资本化吗？','CAS-06-2006-A008','研究阶段支出发生时计入当期损益|不能仅凭未来成功预期资本化')
add('按2006年无形资产准则，开发阶段只证明技术可行就够资本化了吗？','CAS-06-2006-A009','不够，五项条件需同时满足|包括完成意图、经济利益方式、资源支持和可靠计量')
add('按2006年无形资产准则，无法预见获益期限的软件资产需要按固定年限摊销吗？','CAS-06-2006-A016 CAS-06-2006-A019','无法预见经济利益期限属于使用寿命不确定|使用寿命不确定不应摊销')
add('按2006年资产减值准则，固定资产减值确认后行情回暖，可以转回吗？','CAS-08-2006-A017','该准则下资产减值损失确认后不得转回|不能将此结论扩展为所有其他准则的资产都不得转回')
add('按2017年收入准则，确认收入的核心是开票还是客户取得控制权？','CAS-14-2017-A004','履行履约义务、客户取得商品控制权时确认|控制权包括主导使用并获得几乎全部经济利益|不能仅因开票确认')
add('按2017年收入准则，客户一边接受服务一边消耗服务利益，属于时点还是时段履约？','CAS-14-2017-A011','满足时段履约第一项条件|需基于客户在履约同时取得并消耗经济利益这一事实')
add('按2017年收入准则，代第三方收款和预计退给客户的钱算交易价格吗？','CAS-14-2017-A014','不计入交易价格|作为负债处理')
add('按2017年收入准则，平台只是代理人，收入应该按总收款还是佣金确认？','CAS-14-2017-A034','需按转让前是否控制商品判断身份|代理人按预期有权收取佣金或手续费确认|不能只看合同名称')
add('按2017年收入准则，付给客户的钱是否总要冲减收入？','CAS-14-2017-A019','一般冲减交易价格|购买客户其他可明确区分商品是例外|超过该商品公允价值部分仍冲减交易价格')
add('按2006年收入准则，货物已经发出但主要风险报酬没有转移，可以确认销售收入吗？','CAS-14-2006-A004','不能仅因发货确认|五个确认条件需同时满足，包括主要风险报酬转移')
add('按2017年政府补助准则，政府无偿提供设备属于政府补助吗？','CAS-16-2017-A002','政府补助包括无偿取得货币性或非货币性资产|需符合政府无偿给予前提')
add('按2017年政府补助准则，日常活动有关的补助一定要进营业外收入吗？','CAS-16-2017-A011','相关的按实质计入其他收益或冲减相关成本费用|无关的计入营业外收支')
add('按2006年借款费用准则，只拿到贷款但项目还没有开工，可以开始资本化吗？','CAS-17-2006-A005','三个条件须同时满足|资产支出发生、借款费用发生、必要购建生产活动开始')
add('按2006年借款费用准则，非正常停工连续4个月，利息怎么处理？','CAS-17-2006-A011','非正常中断且连续超过3个月应暂停资本化|暂停期间计入当期损益直到恢复|达到预定状态必要的正常程序不适用该暂停结论')
add('按2006年借款费用准则，项目已经达到预定可使用状态但还没办竣工决算，利息是否继续资本化？','CAS-17-2006-A012','达到预定可使用或可销售状态时停止|之后借款费用发生时确认为费用|不能仅以未决算延续资本化')
add('按2006年借款费用准则，专门借款当期利息30万元，未动用资金利息收入5万元，其他资本化条件满足时净额是多少？','CAS-17-2006-A006','专门借款利息减未动用资金收益|净额25万元|结论依赖题设资本化条件','calculation')
add('按2018年租赁准则，承租人是否要对所有租赁都确认使用权资产和租赁负债？','CAS-21-2018-A014 CAS-21-2018-A032','一般在租赁期开始日确认|短期及低价值租赁可选择简化处理|选择简化时按直线或其他系统合理方法计入成本或损益')
add('按2018年租赁准则，租赁负债本金和利息支付在现金流量表归入哪里？','CAS-21-2018-A053','偿还租赁负债本金和利息属筹资活动流出|简化短期低价值租赁及未纳入负债的可变付款属经营流出')
add('按2006年会计政策和估计准则，修改设备预计使用寿命导致的估计变更，要追溯重算以前年度吗？','CAS-28-2006-A009','会计估计变更用未来适用法|影响当期或未来期间按对应期间确认|题设应确实属于估计变更')
add('按2006年资产负债表日后事项准则，非调整事项应改动期末报表金额吗？','CAS-29-2006-A006','不应调整资产负债表日财务报表|本条不代表免除其他条款规定的披露义务')
add('按2006年现金流量表准则，所有短期投资都算现金等价物吗？','CAS-31-2006-A002','需同时具有期限短、流动性强、易于转换为已知金额现金、价值变动风险很小|不能仅以短期认定')
add('按2006年关联方披露准则，两家公司仅仅都是国有企业就一定属于关联方吗？','CAS-36-2006-A006','仅同受国家控制且不存在其他关联关系时不构成关联方')
add('按2017年持有待售准则，管理层口头想卖设备就能分类为持有待售吗？','CAS-42-2017-A006','还需当前状况可立即出售|出售极可能发生，包括决议、确定购买承诺、通常一年内完成|所需审批应取得')
add('按2014年基本准则，计划明年买入的设备今天能确认为资产吗？','CAS-basic-2014-A020','未来预期交易不形成资产|资产来源于过去交易或事项')
add('按2014年基本准则，为了利润稳定，可以随意把本期交易推迟确认吗？','CAS-basic-2014-A019','不得提前或延后|及时确认已发生交易或事项')
add('按2014年基本准则，未拥有法律所有权的资源是否必然不能作为资产？','CAS-basic-2014-A020 CAS-basic-2014-A021','可拥有或控制，并非必须拥有所有权|还需经济利益很可能流入及成本或价值可靠计量')
add('按2006年存货准则，存货成本包含哪些大类？','CAS-01-2006-A005','采购成本、加工成本和其他成本')
add('按2006年固定资产准则，设备预计使用和处置都不能产生经济利益，还应保留确认吗？','CAS-04-2006-A021','满足终止确认条件|不得凭未知信息编造处置损益金额')
# 16 adjacent-version comparisons, including two pairs for CAS37.
families=defaultdict(list)
for d in docs: families[d['standard_id']].append(d)
for family, versions in sorted(families.items()):
    versions.sort(key=lambda d:d['version_year'])
    for old,new in zip(versions,versions[1:]):
        old_a={a['article_number']:a for a in by_doc[old['id']]}
        changed=[a for a in by_doc[new['id']] if a['article_number'] in old_a and a['article_number']>=2 and re.sub(r'\s+','',a['text'])!=re.sub(r'\s+','',old_a[a['article_number']]['text'])]
        a=next((a for a in changed if len(a['text'])<1800 and len(old_a[a['article_number']]['text'])<1800),changed[0])
        b=old_a[a['article_number']]
        add(f'对照附件《{old["title"]}》和《{new["title"]}》，分别说明两版{a["article_label"]}写了什么、有哪些差异。条号相同是否就能认定主题完全对应？',[b['id'],a['id']],['分别按文件年份引用两条原文','指出文本差异；编号相同不保证主题相同','不要将两版拼成一条规则，也不要据此断定目前适用版本'],category='version_comparison',origin='template_from_verified_text_difference')
# Cross-clause/standard tasks with explicit versions and supporting evidence.
add('按2006年存货和资产减值准则，存货跌价与固定资产减值都能在价值恢复后转回吗？','CAS-01-2006-A019 CAS-08-2006-A017','存货在原计提范围内可转回|资产减值准则下损失不得转回|分别引用，不能推广单一规则','multi_evidence')
add('按2006年无形资产准则，研究与开发阶段的支出确认有什么不同？','CAS-06-2006-A008 CAS-06-2006-A009','研究阶段计入当期损益|开发阶段五条件同时满足才确认无形资产','multi_evidence')
add('分别依据2006年与2017年收入准则第四条，销售收入确认的核心表述有何差别？','CAS-14-2006-A004 CAS-14-2017-A004','2006年列出主要风险报酬转移等五条件|2017年强调履约义务和客户控制权|不要混合为同版规则','multi_evidence')
add('按2006年借款费用准则，开始资本化、暂停资本化和停止资本化分别看什么条件？','CAS-17-2006-A005 CAS-17-2006-A011 CAS-17-2006-A012','开始需三个条件同时满足|暂停为非正常中断连续超过3个月，有正常程序例外|达到预定状态停止','multi_evidence')
add('按2018年租赁准则，短期和低价值租赁简化处理与一般租赁的确认、现金流分类有什么不同？','CAS-21-2018-A014 CAS-21-2018-A032 CAS-21-2018-A053','一般确认使用权资产及租赁负债|符合条件并选择简化可不确认|一般本金利息筹资，简化付款经营','multi_evidence')
add('依据2014年基本准则和2006年存货准则，资产确认与存货确认的共同条件是什么？','CAS-basic-2014-A021 CAS-01-2006-A004','经济利益很可能流入|成本或价值能够可靠计量|基本准则还以符合资产定义为前提','multi_evidence')
add('按2017年收入准则，如何区分收入确认时点与收入计量金额？','CAS-14-2017-A004 CAS-14-2017-A014','确认依据履约和控制权|计量采用分摊至履约义务的交易价格|代收与预计退款不计入交易价格','multi_evidence')
add('按2006年固定资产和会计估计准则，折旧所依赖的估计改变后应如何理解处理期间？','CAS-04-2006-A014 CAS-28-2006-A009','折旧基于应计折旧额等估计|确属估计变更采用未来适用法|不能无事实地认定为前期差错','multi_evidence')
add('依据2006年日后事项准则，调整事项与非调整事项的判断核心是什么，非调整事项是否调报表？','CAS-29-2006-A002 CAS-29-2006-A006','调整事项为期末已有情况的新证据|非调整事项为期末后发生情况|非调整事项不改期末报表','multi_evidence')
add('按2014年基本准则，收入与所有者投入资本相同吗，所有者权益如何定义？','CAS-basic-2014-A026 CAS-basic-2014-A030','收入与所有者投入资本无关|所有者权益为资产扣除负债后的剩余权益','multi_evidence')
# Missing version or transaction facts: deliberately no single gold article.
for q,groups,points in [
 ('收到货款就能确认收入吗？',['14'],'先问适用版本及合同履约、控制权等事实|不能仅因收款确认'),
 ('我们这份租赁可以不上资产负债表吗？',['21'],'先问适用版本、承租或出租角色、租期和资产价值等|不能直接豁免'),
 ('政府打来的钱全部计入营业外收入对吗？',['16'],'询问适用准则版本及款项性质、日常活动关联|不能全部归为营业外收入'),
 ('我们做了一笔债务重组，收益是多少？',['12'],'需版本、债务账面价值、对价和重组方式|资料不足不编金额'),
 ('企业互换设备是不是都按公允价值入账？',['07'],'需适用版本、商业实质、公允价值可靠性等|不无条件认定'),
 ('这个金融资产应该按什么类别核算？',['22'],'需版本、合同现金流特征和管理业务模式等|不能凭一句话归类'),
 ('保险合同应按哪一年那版规则做账？',['25'],'需要报告期间、企业适用情况与正式实施依据|不能从附件最大年份推断'),
 ('长期股权投资都用成本法吗？',['02'],'需版本与控制、共同控制、重大影响等关系|不能全部用同一方法'),
 ('我们要编合并报表，所有持股公司都合进去吗？',['33'],'需适用版本及控制等具体关系|不能只凭持股一概合并'),
 ('这笔费用是会计政策变更还是估计变更？',['28'],'需要变更事项、原因与原处理信息|信息不足先澄清')]:
    add(q,[],points,'clarification','clarify',groups)
# Corpus/tool boundaries and temporal uncertainty.
for q,groups,points in [
 ('请按附件给出2026年最新且完整的收入准则全文。',['14'],'附件无法证明2026年最新完整文本|需要核验官方现行版本，不编造'),
 ('请给出本次附件所有文档的财政部原始下载链接。',[],'上传文件未提供已验证官方链接|不能捏造URL'),
 ('根据这些准则查出我公司2025年审计净利润。',[],'缺公司报表及审计数据|准则不是公司经营数据'),
 ('请按附件告诉我今天人民币兑美元即期汇率。',['19'],'无实时行情数据或工具|不能编造汇率'),
 ('今年企业所得税优惠税率是多少？',['18'],'所得税会计准则不等于完整税收优惠法规|附件不足以确定现行税率'),
 ('请直接替我把财务报表报送到税务局。',[],'没有申报工具或授权|不能声称完成报送'),
 ('附件里的2006年版本现在是不是全部失效？',[],'仅凭文件年份不能判断全部现行效力|需要正式废止或衔接依据'),
 ('请依据附件提供美国ASC 606的英文全文。',['14'],'附件为中国准则，不包含ASC606全文|不能冒充原文'),
 ('这些材料能证明我公司一定符合上市财务条件吗？',[],'缺少公司财务及适用上市规则|不能作确定结论'),
 ('请告诉我附件没有的2025年新准则条文号和具体内容。',[],'不能为缺失文档编造条文|请求提供来源')]:
    add(q,[],points,'unanswerable','abstain',groups)
# Group all versions of each standard and cross-standard questions together.
parent={d['standard_id']:d['standard_id'] for d in docs}
def find(x):
    while parent[x]!=x:
        parent[x]=parent[parent[x]]; x=parent[x]
    return x
for r in rows:
    gs=r['standard_families']
    for g in gs[1:]: parent[find(g)]=find(gs[0])
components=defaultdict(list)
for r in rows:
    gs=r['standard_families']; key='family:'+find(gs[0]) if gs else 'boundary:'+r['id']
    components[key].append(r)
# Stable grouped split (not random paraphrase split). Whole-family holdout.
for key,rs in components.items():
    split='test' if int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%10<3 else 'dev'
    for r in rs: r['split']=split; r['split_group']=key
for split in ('dev','test'):
    subset=[r for r in rows if r['split']==split]
    (OUT/f'questions.{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in subset),encoding='utf-8')
(OUT/'questions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
report={'questions':len(rows),'categories':dict(Counter(r['category'] for r in rows)),'splits':{s:{'total':sum(r['split']==s for r in rows),'categories':dict(Counter(r['category'] for r in rows if r['split']==s))} for s in ('dev','test')},'coverage_documents':len({by_id[e]['document_id'] for r in rows for e in r['evidence_ids']}),'annotation_status':'draft_needs_domain_review','split_policy':'whole standard-family connected components, all versions and cross-standard questions in same split','benchmark_scope':'synthetic corpus-derived development dataset; not independently human-authored or an untouched external benchmark'}
(OUT/'questions_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
