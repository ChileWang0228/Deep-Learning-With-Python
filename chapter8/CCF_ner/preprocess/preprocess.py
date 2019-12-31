import pandas as pd
import codecs
import re
import json
import sys
sys.path.append("/home/wangzhili/chilewang/clean_ccf_ner")  # 添加项目根路径，避免在服务器上调用代码时找不到上一级目录的模块
from config import Config

"""
按照标点符号切割的预处理数据
"""
config = Config()
len_treshold = config.sequence_length - 2  #  每条数据的最大长度, 留下两个位置给[CLS]和[SEP]
data_dir = config.new_data_process_quarter_final
print(data_dir)

# 原始数据集
train_df = pd.read_csv(config.source_data_dir + 'Round2_train.csv', encoding='utf-8')
test_df = pd.read_csv(config.source_data_dir + 'Round2_Test.csv', encoding='utf-8')

# 找出所有的非中文、非英文和非数字符号
additional_chars = set()
for t in list(test_df.text) + list(train_df.text):
    additional_chars.update(re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', str(t)))

# 一些需要保留的符号
extra_chars = set("!#$%&\()*+,-./:;<=>?@[\\]^_`{|}~！#￥%&？《》{}“”，：‘’。（）·、；【】")
print(extra_chars)
additional_chars = additional_chars.difference(extra_chars)

def stop_words(x):
    try:
        x = x.strip()
    except:
        return ''
    x = re.sub('{IMG:.?.?.?}', '', x)
    x = re.sub('<!--IMG_\d+-->', '', x)
    x = re.sub('(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', x)  # 过滤网址
    x = re.sub('<a[^>]*>', '', x).replace("</a>", "")  # 过滤a标签
    x = re.sub('<P[^>]*>', '', x).replace("</P>", "")  # 过滤P标签
    x = re.sub('<strong[^>]*>', ',', x).replace("</strong>", "")  # 过滤strong标签
    x = re.sub('<br>', ',', x)  # 过滤br标签
    x = re.sub('www.[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', x).replace("()", "")  # 过滤www开头的网址
    x = re.sub('\s', '', x)   # 过滤不可见字符
    x = re.sub('Ⅴ', 'V', x)

    for wbad in additional_chars:
        x = x.replace(wbad, '')
    return x

train_df['text'] =  train_df['title'].fillna('') + train_df['text'].fillna('')
test_df['text'] =  test_df['title'].fillna('') + test_df['text'].fillna('')

# 清除噪声
train_df['text'] = train_df['text'].apply(stop_words)
test_df['text'] = test_df['text'].apply(stop_words)
train_df = train_df.fillna('')


"""
找出错误标签
"""
label_list = train_df['unknownEntities'].tolist()
text_list =  train_df['text'].tolist()
id_list =  train_df['id'].tolist()
false_get_id = []
false_get_label = []
for i, label in enumerate(label_list):
    text = text_list[i]
    idx = id_list[i]
    l_l = label.split(';')
    not_in = []
    for li in l_l:
        if li not in text:
            not_in.append(li)
    if len(not_in) > 0:

        false_get_id.append(idx)
        false_get_label.append(label)


"""
修复错误标签
"""

repair_id_label = ['大象健康科技有限公司;健康猫', '人人爱家金融', '速借贷;有信钱包', '速借贷;有信钱包', '速借贷;有信钱包',
                   '速借贷;有信钱包', '软银;(必发)BETFAIR;火币', 'ATC国际期货;香港恒利金业;嘉信金服', 'Finci芬吉', '闪电借款;掌众财富',
                   '陀螺', '', '', '大象健康科技有限公司;健康猫', '宝点网;e租宝',  '大象健康科技有限公司;健康猫', '大象健康科技有限公司',
                   '', 'Exp金融资产;expasset项目;Plustoken钱包;expasset;Exp亚斯特', '钱-宝-网', 'Vpay支付', '', '盈康易和元消痛贴',
                   'Bitfund', 'GEC币', '时代证卷', '火星钱包;火星数字资产银行', '中汇国际期货;中宏资管;先汇国际', 'ST国际（搜宝国际）',
                   '和信售货机限公司', '洛安期货;昶胜国际;BKB数字货币;中恒策略;MBGMarkets', '', '', '', '沪深理财;首投理财',
                   'GEC币', '', '安盛', '', '智慧晶商城', '正宇控股', 'ShapeShift;Coinbase;CactusCustody;GenesisCapital;贝宝金融;比特大陆;Matrix;Matrixport;BitGoInc.;MatrixportBeta',
                   'wotoken', 'wotoken;WoToken', 'wotoken;Wotoken', 'wotoken;WoToken', 'wotoken;WoToken', '小微E贷通', 'ECOIN', 'GEC环保币',
                   'PTFX普顿外汇;PTFX;PTFX（普顿）;聚宝金融;马胜金融;3M;IGOFX;金殿环球;OCTO;澳沃国际;期海聚金;期海财经', '一加二金融;富车在线;万贝贷;牛娃互联网金融;古德金融超市',
                   'JOJOMARKETS', '团贷网;钱香金融;金沙江;点亮;磐庆;玖臻资本;PPmoney;爱钱进;熊猫金库;饭饭金服;惠人贷;拍拍贷;麻袋财富', 'EOS',
                   'ATC艾特币;ZFB-致富链;趣步;巴特币;TBC天宝币;艾特币;智慧晶',
                   '有融网;银谷财富;爱福家;银豆网;长沙赛鼎生物科技有限公司;金瑞龙理财;湖南振湘网络科技有限公司;伊思多尔', '乐易Bank', '快刷;支付通;金中宝',
                   '应价零批;Lnko;plustoken;金网安泰;信雅达;PlusToken;lnko',  '普信金服APP;宏利保险;北京征和惠通基金管理有限公司;征和惠通;信中利;征和控股集团', '普信金服APP',
                   '普信金服APP', '普信金服APP', '东霖国际;波场超级社区;Plustoken;PlusToken;GCG钜富;PTFX普顿;RCFX;GCG钜富金融;GCGASIA鉅富金融;GCGASIA;香港钜富金融公司;外汇米治;SOXEX交易所;闪链SHE;火币HT',
                   '北斗股权;北斗生态圈;北斗期权;易购商城;趣步;循环信用卡', 'MGCTOKEN;mgctoken;MGCtoken',  '安徽天策;Plustoken;Lnko钱包;SOXEX交易所;波场超级社区;PlusToken;米治;PTFX',
                   '淘优乐;链豆', '', '雅布力', '友信证券;友信智投', '', '华盈城市集团;厦门中海航集团;恒优国际集团', '图灵资本;道生资本;小智投资;银杏谷资本', '',
'金蛋理财;网利宝;金百万;大麦理财;伽满优;玖富;米族金融;合众e贷;拼客顺风车;麦麦理财;盘龙财富;翼龙贷;证券单;微贷网;指旺财富;银谷在线;华侨宝;投哪网;团贷网;陆金所;积木盒子;悟空理财;广信贷;向上金服;银多网;花生米富;用钱宝;乾包网;盈盈理财;海鹭金服;智新宝;京贷金服;点牛金融;融信网;易享贷;51人品;金融工厂',
                   'UBank', '永利金控控股有限公司;UBS香港', '金机宝', '3m3.0互助;mmm3.0', '红马优购;车智汇狗狗币;酒链世界;霍特币;趣步;dogx钱包;以太坊;闪电鸡',
                   'ALicoin;TheDAO项目', '四川创梦森林软件科技有限公司;Cmsl', 'FCoin;火币', 'DEtoken', 'dogx;dogxwallet', 'DragonEx龙网;IOST',  'EXPASSET;EXP', '利润钱包;EXP-ASSET', 'FCoin',
                   'Finci芬吉;嘉实资本;TF环球金融;众元国际;捷盛国际;金源财富;GMO奥诺;鼎盈信投;中恒策略;億運富國際;速汇国际;辉立国际;世纪金业;艾斯国际;金山金融;MALAFY;宏源国际;鸿安资本;DGINT;问鼎财富;鸿昇国际;鸿翔国际;圆汇金融;唐印国际;恒牛策略;东财国际;创昇国际;锦丰国际',
                   'GEC', 'GEC', 'GEC', '瑞波(XRP)', 'HES和氏币', 'iBank数字钱包', 'IOToken(米奥钱包）;IOToken', 'JMC', 'Jojomarkets', 'KB数字资产兑换平台', 'LongBit',
                   'LXFX;丰盈服务;立盟策略;易配配资;捷盈资本;众昇策略;boss金服', 'DLC数字货币;MCC数字矿工币', 'MChain;EXX交易所;蜂鸟财经;道轩资本', 'Mchain;MChain', 'PlusToken',
                   'ProsperToken;LTC莱特币;EOS柚子;狗狗币;BCH比特现金;XRP瑞波币;osp;达世币;DOGE狗狗币;火币;plustoken;ETC以太经典;DASH达世币', 'SKY;sky云世纪', '华润商业;中星集团;金丰投资;上海古北集团',
                   '速通宝Vpay;速通宝;瑞波币', 'WAD国际拆分理财平台','WAD国际金融平台', 'WBEX', 'WPAY', 'YouBank', 'jojomarkets', '惠恩商城;MGC钱包;东霖国际', '百信通国际;CITIGLOBAL花旗国际;海.贝国际;海慧通;众元国际;恒牛策略;世纪金业;诺安期货;金源财富;白象国际;辉立国际;HATSKY;博迈在线;bitkoc德林资本;艾斯国际;JTL国际;长虹资本;HDI国际;嘉晟财富;SpeedTrad速汇国际;mrt捷盈资本;万鼎国际;信融期权;恒利金业;britrading;新源财经;东吴金服;创昇国际;CXM希盟;宏源国际;旭升策略;富通国际;海利国际;合欣国际;东财国际;九梦财富;中赢国际EGML环球金融;国人策略;优信外汇;汇丰联合;鸿运信投;鼎盈信投;信邦策略;宏观策略;聚富策略;汇融国际',
                    '北京丽泽金融商务区控股有限公司;北京金唐天润置业发展集团;北京戴德梁行物业管理有限公司', 'Drivechain;侧链(Sidechain)', '58coin', 'OK链', '', 'Vpay;VPAY支付', '',
                   '菠菜理财;上海犇序实业有限公司;象象财富;小灰熊金服;云端金融;掌悦理财;央金所;巨人理财;利利金服', 'Guardian', '钜派投资集团;布尔金融', '环球金融;华宝基金', '成都奇米达',
                   '成都潜隆贷;麻袋理财;君融贷;沃时代;金银猫;恒信易贷;联连理财', '成都潜隆贷;宝通网;汉金所;鑫鹏贷;中旭鸿基投资;丰鼎金融;联鼎投资;昊祥投资;国有投资展恒理财',
                   '捷信（中国）金融公司;捷信', '侠侣联盟厦门侠网旅游服务有限公司;厦门侠网旅游服务有限公司;旅划算广州市旅划算国际旅行社有限公司',  '承返网;承返网(承返(广州)网络科技有限公司)',
                   '合拍贷;春天金融;稳银在线;可易金融;起点贷;深圳盈泰联合投资管理有限公司;深圳智鼎投资管理有限公司;深圳市中云金融服务有限公司;小宝金服;盈泰联合;印子坊;浙江颐荣资产管理有限公',
                   '维卡币', '优客工场;人人车;唱吧;考拉先生;蔚来汽车;雷蛇;柔宇科技;汉富资本', '', '金证股份;欧普康视', 'ORANFLAG', '旭隆金业;旺润配资;金银策略;汇丰鸿利;云旗金服;策略资本;翻翻配资;中首上上策;震泰国际;亨达国际;稳赢策略;四川大宗;海牛金服;指南针大宗;盈策略;牛360配资;一股期权;维海配资;银华中期;博时国际期货;中泰期权;创利融;步步盈配资;中证期权;航心配资;鼎鑫金业;创盈金服;九五配资;鼎牛配资;亿配资;华瑞国际;鼎盛配资;艾德配资;百汇期权;点点金富通国际',
                   'sumtoken;fCOIN交易所', '', 'PPmoney;爱钱进;轻易贷;永银贷;一诚一贷;一人一贷', '东霖国际', '', '阿尔泰平台', '商房所;捷麦理财;易麦理财;砚下金融;俊掌柜;91飞猫;外快理财', 'DFC;东方城', '民生证券;中天金融;民金所;泛海控股;亚太财险',
                   '华远国际;易资配;速汇国际;汇丰联合;泛金国际;信投在线;EGML环球金融;创远世纪金融;东吴金服;豪根国际;AJPFX;中港国际;股融易;信邦策略;大彩配资;飞客在线;世纪金融;世纪集团金业控股有限公司;金景配资;中瑞财讯;西安环海陆港商品交易中心;方正国际;新源财经;50ETF;弘基金融;富通国际;恒信财富;日照大宗商品交易所;海慧通;洪富期货;创昇国际期货;海拓环球融资融券;安信金控',
                    '长颈鹿;MT', '中信资本;中信资本;凤凰智信;凤凰金融;东方资产',
                   '高胜投资;投行配资;九鼎金策;香港信诚资产;中鑫建投;帝锋金业;向上金服;银丰配资;粤友钱;策马财经;盈龙策略', # 2551
'人人贷;钱来也;帛扬集团', '零花钱;趣妙租;亚远科技', '一直牛',
'钱宝科技;招钱进宝;秒到账;钱宝', '苏宁体育;苏宁文创', '布尔金融;冠联投资家园.财富领航',
'益冠创投;金凤凰;煌萨投资;国富通;易钱汇;豆包金服;观金鉴股;浩然众筹;房金网;金豆包;91飞猫;朔漪众筹;穆金所;中信创;好牛创投;汇信聚业;黎明国际;易众网;吆鸡理财', '宜聚网;和信贷;恒易融;花虾金融;恒慧融;财富中国;道口贷;信融财富;东创在线;小赢理财;草根投资;信广立诚贷;聚财猫;善林金融;宁波东创投资管理有限公司;东创投资;多乐融;金瑞龙;财富星球;米庄理财;达人贷;信而富;洋钱罐;今日捷财;财猫;短融网;银豆网;恒昌;钱盆网;笑脸金融;云端金融;开鑫金服', '海航集团;航海创新', 'HIIFX海汇国际', '欧莱雅;嗨团团购', '红域短视频;火牛视频', '快易点;八里香', '陆金所;小牛在线;蜂融网;平安集团', '钱牛牛;麦子金服;互融宝;51人品;拓道金融;玖富;付融宝', '花生日记', '乾包网;津启宝;九药网;好友邦金服;点聚财;生意贷;聚财;金羊金服;东上金服;中航生意贷;邦金服', '和信贷;贯通金服;普汇云通;点融网;众筹平台;中融民信;酷盈网;点融;盎盎理财;众力金融', '积木盒子;全球金融;网贷;宜人贷;人人贷;巴巴汇', '中国保险;加油宝;同花顺财经;合时代', 'FCoin', '海拓环球;融资融券', '链上钱包;以太森林', '新橙分期;新口子;今日推荐', '金殿环球', '金包豆;“兄弟”车贷;昌和财富;昌久财富;聚利众筹;建军财富;聚汇天下;诚天财富;金贝贝;桑善金服;金致财富;絮东投资;建元投资;豆蔓智投;中智魔方;亦川资本;红丰智投;富捷金服;厚元投资;紫檩金融;鸿百佳投资;酷盈网;中仁财富;理财咖;益冠创投;华隆资产;涌集投资;聚金袋子;聚乐资本;博美投资;融创嘉诚;大盈投资;金苏在线', '金开贷',
                   '锦安财富;金色木棉;北京卡拉卡尔科技公司;ST德奥;意隆财富', '信邦策略;中赢国际;花旗;贵金属;恒牛策略;现货白银;金源财富;海利国际;嘉晟财富;澳大利亚证券及投资委员会（ASIC）;新源财经;炒外汇英国金融;聚富策略;富通国际;速汇国际;白象国际;英国金融行为管理局（FCS）;瑞士金融市场监督管理局（FINMA）;速赢;新西兰金融市场管理局（FMA）;外汇投资;恒利金业;汇丰;环球;优信外汇;新华富时;环球金融;汇融国际', '天安金交中心;厦金中心;厦金理财平台;天安（贵州省）互联网金融资产交易中心股份有限公司;北京鑫旺阔达投资有限公司;鑫旺投资;深圳市景腾瑞贸易发展有限公司;厦门国际金融资产交易中心有限公司', '移动钱包;玖富数科集团;玖富钱包',
'分期乐;玖富普惠;投哪网;小赢理财;微贷网;桔子理财;玖富钱包;网上赚钱;小赢钱包;东方证券;玖富;贷网;龙支付;宜人贷;麻袋财富;国信证券',
'分期乐;宜人贷;玖富普惠;投哪网;小赢理财;微贷网;桔子理财;玖富钱包;网上赚钱;小赢钱包;东方证券;玖富;龙支付;微贷;国信证券;麻袋财富',
'酒业链wnn', '中资信汇投资打点有限公司', '广发基金;大成基金;博时基金;鹏华基金;汇添富基金;jojomarkets', '汇置投资;挑战者;汇置财富',
'凯顿', '皇玛金融;中北选买;中远期货;创远世纪金融;海利国际;青岛西海岸;恩圣威ncy;新源财经;华远国际;聚富策略;AJPFX;富通国际;速汇国际;丰盛金融;艾利威国际;保诚国际;豪根国际;游资通;方正国际;恒信财富;粒子金融;恒利金业;泛金国际;优信外汇;创远世纪;世纪金融;汇融国际;中首投资;安信;中融金业',
'新华都;三江购物', '', '瑞波币;恒星币;万维币;通盛币;珍宝币;富豪币;万福币;吉祥币;视界链;农业链;天使链;流量魔石;金元币;西游链;高兴币;电能链;lmc柠檬币;kdc凯帝币;csc炭汇币;scc足球币;绿链;acc防伪币;fyb弗益币;汇择投资;正谦益;睿鑫宝;德爱社区;微韵文化;益路同行;山海经;融易通;特色三妹;至尊;阿川;星火草原;恩威商城;CNY金融互助;公益社区掌心众扶;友钱宝;友义宝;影子银行;智富宝;云支付|云付通;精神传销心灵培训;亿加互助;ICA;微韵文化;kdc凯帝币;流量魔石;云支付|云付通;fyb弗益币;智富宝;星火草原;万福币;德爱社区;公益社区掌心众扶;天使链;scc足球币;影子银行;lmc柠檬币;富豪币;睿鑫宝;友义宝;吉祥币;恩威商城;CNY金融互助;acc防伪币;正谦益;金元币;友钱宝;维卡币;ICA;绿链;特色三妹;恒星币;视界链;至尊;阿川;精神传销心灵培训;益路同行;csc炭汇币;农业链;珍宝币;融易通;汇择投资;通盛币;马克币;山海经;亿加互助;西游链;万维币;高兴币;电能链',
'坤吉国际;EQR;FDEX;贵州国际交易中心;广州西勃商品交易中心;创昇国际期货;点牛融资融券;信捷策略;众昇策略;时盛财汇;鼎点策略;钱盈配资;国金策略;嘉露国际;有富策略;东方汇盈;小金橘策略;红牛策略;真牛科技;中航江南期货;象屿期货;贝赢网;信诚资产;涨悦财金;海慧通;壹恒国际;鼎点策略;钱盈配资;顺通在线;国金策略;嘉露国际;迅视资管;期权专车;桑杰股权;花旗资本;BKB;九州金服;中盟国际;中浙金控海博利;创辉国际;海南大宗;诺安期货;路易泽;安信金控;百益策略;期货大赢家', '', '易金融;麻袋财富', '理理财', '立刷', '钱保姆;分秒金融;饭饭金服;掌悦理财;黎明国际;金汇微金;一点金库;金统贷;有融网', '零点矿工', '招财宝;香港安盛投连险;AsiaOne', '平安银行;江苏银行;蚂蚁财富;天天基金;陆金所;微交易;嘉合基金', '',
'觅信DEC', '积木盒子;钱来也;钱来也网络借贷;你我贷',
'诺德基金;中天证券;大同证券;山西证券', '广州承兴营销管理有限公司;诺亚财富',
'恒利金业;FusianGallant;香港富赢通;高盛亚太;菲特尔国际;粒子金融;格兰特;中泰期权;TRENDS;欧克斯;中首投资;优越投资;嘉晟财富;速汇国际;青岛西海岸;豪根国际;保卓国际;Morse;火币网;奥瑞国际;琥珀帝国;大赢家;星亘国际;富时罗素;嘉兰待;法纳金融;高盛亚太;大赢家期货;香港优越投资中心;中元天颐;中远期货;香港富盈;宝丰国际;火币网;CFEX;帕克金融;鼎和金控;火币;中浙金控;彭义昆维权;富通国际;世纪金业;ATTEX', '领航国际资本', '前金融;陆金所;链链金融;房金所;中金贷;津融贷;车赚', '马胜金融;普顿PTFX;普顿ptfx;聚宝金融;IGOFX;3M;PTPrutonMegaBerjangka;PrutonCapital', '沃尔克外汇;MMM金融互助;马胜金融;亨英集团;HIIFX海汇国际;EA智能交易;IGOFX;HYBX', '红威投资;金融网;富利宝;豫之兴资本;钱富通;竞优理财;丰鼎金融;真信汇通;金财富;智慧理财;聚财;全局金服;榕巨互金;信息网;天诚财富;星通财富;梵丰投资;国有投资;德众金融;吉盟财富;永利宝;亿企聚财;花橙金融;理财网;吉农投资;储信理财;火理财',
 '合创', '光子链PTN', '京东;苏宁;唯品会;淘宝;考拉;权健;华林', '京东;苏宁;唯品会;淘宝;考拉;权健;华林', '天猫;淘宝;京东;优品汇;全返通',
 '融创中国;摩根士丹利;德意志银行;农银国际', 'e租宝;钱宝网;江苏联宝', 'BitMEX', 'GEC',
 '夏商风信子;时福全球购;恒优国际;宝象商务中心;金淘惠源;沃洋优品;酩悦酒业;欧颂酒业;优传保税;跑街',
 '淘宝;京东;网易', '东方证券股份有限公司;花旗环球金融（亚洲）有限公司',
 '神州泰岳投资;米庄理财;神州泰岳;图腾贷;沃时贷;米庄理财', '', 'A股头条', '',
 '速汇国际;SpeedTrad;新华富时A50;德指DAX30;恒指HSI;泰达币', '速通宝vpay',
 '悟空理财;晋商贷;铜板街;泰然金融;玖富;融牛在线;可溯金融;微贷网',
 '比特币;莱特币;无限币;夸克币;泽塔币;烧烤币;隐形金条', '优乐商城;淘优乐',
 '米缸金融;富管家;鑫聚天利;富盈;金理财;宁富盈;天安（贵州省）互联网金融资产交易中心股份有限公司;北京航天浦盛科技中心;天安金交中心',
 '富管家;鑫聚-天利;天安金交所', '寒武创投;熠美投资', '红岭创投;PPmoney;乐享宝', '金融投资;风险投资;钱生钱',
 '莱次狗;摩拜链(MobikeChain);以太坊;遨游;共生币;CNHKC;CEC;ENE;共生币;遨游;摩拜链', '挖易',
 '玖富数科;友信智投;普惠金融;金融科技;宜人金科;友信金服', '雷达币;雷达钱包;雷达支付', '未来星球', 'jojomarkets',
'西投控股;西安经开城投;西安城投(集团);西安曲江文化;长安信托;西安金控',
'远特通信;远特喜牛;远特;喜牛', '华霖财富管理股份有限公司;华霖金服',
'钱宝;百川币;e租宝;圣商;中晋;e租宝;泛亚;鲜生友请', '伽满优;富友支付;乾包网',
'伽满优;富友支付;乾包网', '伽满优;富友支付;乾包网', '策略通;牛股盈;新纪元期货;花旗证券;广汇大宗商品交易中心;贵州西部农产品交易中心;众生策略;沃伦策;中阳期货;宏琳策略;股亿讯;莱赢宝;金盛商贸;创投大师;芬吉;牛来了;掌互通;鼎盈信投;沪深689策略;国金策略;神圣策略;益达商城;壹恒国际;复兴恒福;神谷策略;江苏百瑞赢;众达国际期货;纯手期货;天兴国际期货;国人期投;超人国际;融盛在线;众赢投顾;神圣策略;股易融;花旗证券;鸿运信投;众生策略;财创期选;劲牛期权;华信策略;中讯策略;创期国投;顺配宝;香港英联策略', 'UnWallet', '小九花花', '众赢;普惠金融;悟空理财;叮当贷;玖富钱包',
 '小象金融;响当当;百仁贷;公众理财;宜泉资本;信而富;99财富;一起理财;酷盈网;人人爱家', '小诸葛金服;芝麻宝金服;天农金融;有融网;狐小狸理财;城城理财;海星宝理财;易纳理财;普益金服;财富中国;小灰熊金服;红八财富;贷你盈;超人贷;台州贷;百金贷;浣熊理财;银号理财;伟民金服;啄米理财;麻宝金服;天农金融;有融网;狐小狸理财;城城理财;海星宝理财;易纳理财;普益金服;财富中国;小灰熊金服;红八财富;贷你盈;超人贷;台州贷;百金贷;浣熊理财;银号理财;伟民金服;啄米理财',
'', '', '中云国际;E路商城', '京东金融;京东理财;小白理财;年年盈;月月盈;金理财;天天盈;币基金', '猎金集团;猎金全民影视',
 '巨人理财;掌悦理财;一点金库;央金所;利利金服;领奇理财;投米乐;领奇理财;一点金库;利利金服;微米在线;掌悦理财;巨人理财;投米乐;微米在线', '宜信惠民投资管理', '以太云', '', '宁波甬坚网络科技有限公司;麦穗金服;钱内助;三金在线;民信金服;利民网;巨如众吧;抢钱通;金投手;壹万木投资;宁海县永坚混凝土有限公司;易麦理财',
 '金证科技;新大陆;兴业数金;工银科技;高伟达;国泰君安研究所;民生科技','仁远资本;贝米钱包;东泽汇顺发;人人贷', '啄米理财;快点理财;甬e贷;多米金融;壹佰金融;津启宝;利魔方;温州贷;招金猫;易贷在线', '富通环球投资;恒信环球投资;恒信国际;恒信贵金属;恒信集团', '五星基金;华安策略', '你我贷;玖富普惠;宜人贷;微贷网', '有信钱包;芝麻分贷款', '花生日记;菜鸟;云集微店;',
 '皮城金融;企查查;海宁民间融资服务中心;海宁皮城', '浙江谢志宇控股集团有限公司;杭州凯蓝汽车租赁有限公司', '蘑菇街;飞猪旅行;侠侣联盟;厦门侠网旅游服务有限公司;厦门侠网旅游服务有限公司', '道琼斯指数;平安证券;HDI', '海贝国际;IGOFX平台', '光大保德信鼎鑫基金;华泰期货',
'MoreToken钱包;Coinone;Tokenstore钱包;BossToken钱包;智能搬砖;BossToken;SecurityToken',
'玖富叮当贷;马上金融;招联好期贷;小鲨易贷', '天津银行;智圣金服;金融理财;全民理财;广州智圣大健康投资有限公司', '智圣金服;金融理财',
'红橙优选;微豪配资;恩圣威;MORSE;易信;新源财经;中北选买;嘉晟财富;恒利金业;ATTEX;优信外汇;速汇国际;威海中元天颐;AJPFX;中元天颐;中泰之星',
'信邦;中赢国际;白象国际;花旗;中瑞财讯;恒利金业;海利国际;嘉晟财富;新源财经优信外汇;环球金融;汇融国际;聚富策略;富通国际;速汇国际', '中信华睿;华安策略;福盛期权;杜德配资;WIRECADD;MALAFY;金田策略;;Helong和隆;银岛配资;世纪金业;鼎盈信投;信融期权;弘基金融;天臣配资;久联优配;致富配资;鼎泽配资;涵星配资;鑫配资;鼎盈信投;信邦策略;百益策略;安信金控;CFX圆汇;格林期货;鸿运信投;信邦策略;宏观策略;金多多配资;罗宾智投;信溢国际;弘基金融;万荣国际;多乾国际;合欣国际;EGML;环球金融;HATSKY;速达国际;中阳期货;丰讯凯国际FDEX',
'华远国际;撮合网;粒子金融;明道配资;长江期货;佳银融资融券;海南大宗商品交易中心;贵州国际商品交易中心;策略资本;稳赢策略;盈策略;川商联宗商品;外汇投资;天元策略;聚富策略;环海陆港;汇融国际;领航配资;新纪元;广州西勃商品交易中心;权金汇;东方财经;中远期货;诚信配资;方正国际;新源财经;艾利威;大连商品交易所;赛岳恒配资;弘基金融;创期国投;盛赢期服', '“慧盈”理财;“家和盈”理财;“增盈”理财', '普信金服APP', '投哪网;麻袋财富;东方证券;桔子理财;微贷网;国信证券;小赢理财;分期乐;宜人贷;小赢钱包', '中金珠宝', 'P2B;微金融;芒果金融', '', '',
'股王配资;DBC币;众融众投;新富金融;恒通国际;微交易;大東方国际商品交易集团;鑫汇环球;大東方国际商品交易集团;恒通国际微交易;DBC币;新富金融;股王配资;众融众投;鑫汇环球',
'中航期货;震泰国际;ainol艾诺;joso聚硕;tfyforex;国峰贵金属',
'plustoken', '亚马逊', 'brt房地产信托', '火币;okex', '嘉盛', '沃客', 'okex', '爱福瑞',
'云讯通;云数贸;五行币;善心汇;LCF项目;云联惠;星火草原;云指商城;世界华人联合会;世界云联;WV梦幻之旅;维卡币;万福币;二元期权;云梦生活;恒星币;摩根币;网络黄金;1040阳光工程;中绿资本;赛比安;K币商城;五化联盟;国通通讯网络电话;EGD网络黄金;万达复利理财;MFC币理财;微转动力;神州互联商城;绿藤理财;绿色世界理财;宝微商城;中晋系;马克币;富迪;万通奇迹;港润信贷;CNC九星;世界云联;沃客生活;天音网络;莱汇币;盛大华天;惠卡世纪;开心理财网;贝格邦BGB;FIS数字金库;SF共享金融;DGC共享币;易赚宝;丰果游天下;天狮集团;薪金融;MGN积分宝;光彩币;亿加互助;GemCoin(珍宝币);老妈乐'


                  ]  # 对应id的修正实体
id_list = train_df['id'].tolist()
label_list = train_df['unknownEntities'].tolist()

for i, idx in enumerate(id_list):
    if idx in false_get_id :
        label_list[i] = repair_id_label[false_get_id.index(idx)]

# 修复过程中漏了几个标签，在这里补上
label_list[2409] = '金融科技（Fintech）'
label_list[2479] = '玖富钱包;玖富数科集团;玖富钱包APP'
label_list[3596] = '盈盈理财;乾包网;臻理财;蜗牛在线'
train_df['unknownEntities'] = label_list
train_df = train_df[~train_df['unknownEntities'].isnull()]  # 删除空标签
train_df.to_csv(data_dir + 'new_train_df.csv')

# 切分训练集，分成训练集和验证集，在这可以尝试五折切割
print('Train Set Size:', train_df.shape)
new_dev_df = train_df[4000:  ]
frames = [train_df[:2000], train_df[2001:4000]]
new_train_df = pd.concat(frames)  # 训练集
new_train_df = new_train_df.fillna('')
new_test_df = test_df[:]  # 测试集
new_test_df.to_csv(data_dir + 'new_test_df.csv', encoding='utf-8', index=False)


def _cut(sentence):
    """
    将一段文本切分成多个句子
    :param sentence:
    :return:
    """
    new_sentence = []
    sen = []
    for i in sentence:
        if i in ['。', '！', '？', '?'] and len(sen) != 0:
            sen.append(i)
            new_sentence.append("".join(sen))
            sen = []
            continue
        sen.append(i)

    if len(new_sentence) <= 1: # 一句话超过max_seq_length且没有句号的，用","分割，再长的不考虑了。
        new_sentence = []
        sen = []
        for i in sentence:
            if i.split(' ')[0] in ['，', ','] and len(sen) != 0:
                sen.append(i)
                new_sentence.append("".join(sen))
                sen = []
                continue
            sen.append(i)
    if len(sen) > 0:  # 若最后一句话无结尾标点，则加入这句话
        new_sentence.append("".join(sen))
    return new_sentence


# 数据切分
def cut_test_set(text_list):
    cut_text_list = []
    cut_index_list = []
    for text in text_list:

        temp_cut_text_list = []
        text_agg = ''
        if len(text) < len_treshold:
            temp_cut_text_list.append(text)
        else:
            sentence_list = _cut(text)  # 一条数据被切分成多句话
            for sentence in sentence_list:
                if len(text_agg) + len(sentence) < len_treshold:
                    text_agg += sentence
                else:
                    temp_cut_text_list.append(text_agg)
                    text_agg = sentence
            temp_cut_text_list.append(text_agg)  # 加上最后一个句子

        cut_index_list.append(len(temp_cut_text_list))
        cut_text_list += temp_cut_text_list

    return cut_text_list, cut_index_list


def cut_train_and_dev_set(text_list, label_list):
    cut_text_list = []
    cut_index_list = []
    cut_label_list = []
    for i, text in enumerate(text_list):
        if label_list[i] != '':
            text_label_list = label_list[i].split(';')  # 获取该条数据的实体列表
            temp_cut_text_list = []
            temp_cut_label_list = []
            text_agg = ''
            if len(text) < len_treshold:
                temp_cut_text_list.append(text)
                temp_cut_label_list.append(label_list[i])
            else:

                sentence_list = _cut(text)  # 一条数据被切分成多句话

                for sentence in sentence_list:
                    if len(text_agg) + len(sentence) < len_treshold:
                        text_agg += sentence
                    else:
                        new_label = []  # 新构成的句子的标签列表
                        for label in text_label_list:
                            if label in text_agg and label != '':
                                new_label.append(label)

                        if len(new_label) > 0:
                            temp_cut_text_list.append(text_agg)
                            temp_cut_label_list.append(";".join(new_label))

                        text_agg = sentence
                # 加回最后一个句子
                new_label = []
                for label in text_label_list:
                    if label in text_agg and label != '':
                        new_label.append(label)
                if len(new_label) > 0:
                    temp_cut_text_list.append(text_agg)
                    temp_cut_label_list.append(";".join(new_label))

            cut_index_list.append(len(temp_cut_text_list))
            cut_text_list += temp_cut_text_list
            cut_label_list += temp_cut_label_list

    return cut_text_list, cut_index_list, cut_label_list


train_text_list = new_train_df['text'].tolist()
train_label_list = new_train_df['unknownEntities'].tolist()
train_id_list = new_train_df['id'].tolist()

dev_text_list = new_dev_df['text'].tolist()
dev_label_list = new_dev_df['unknownEntities'].tolist()

test_text_list = new_test_df['text'].tolist()
test_id_list = new_test_df['id'].tolist()

train_cut_text_list, train_cut_index_list ,train_cut_label_list = cut_train_and_dev_set(train_text_list,  train_label_list)
dev_cut_text_list, dev_cut_index_list, dev_cut_label_list = cut_train_and_dev_set(dev_text_list, dev_label_list)

test_cut_text_list, cut_index_list = cut_test_set(test_text_list)

"""
测试切分是否正确
"""
flag = True

for i, text in enumerate(train_cut_text_list):
    label_list = train_cut_label_list[i].split(';')
    for li in label_list:
        if li not in text:
            print(i)
            print(li)
            print(text)
            flag = False
            print()
            break
        if li == '':

            print(li)
            print(text)
            flag = False
            print()
if flag:
    print("训练集切分正确！")
else:
    print("训练集切分错误！")


flag = True
for i, text in enumerate(dev_cut_text_list):
    label_list = dev_cut_label_list[i].split(';')
    for li in label_list:
        if li not in text:
            print(i)
            print(li)
            print(text)
            print()
            flag = False

if flag:
    print("验证集切分正确！")
else:
    print("验证集切分错误！")

# 保存切分索引
cut_index_dict = {'cut_index_list': cut_index_list}
with open(data_dir + 'cut_index_list.json', 'w') as f:
    json.dump(cut_index_dict, f, ensure_ascii=False)

dev_cut_index_dict = {'cut_index_list': dev_cut_index_list}
with open(data_dir + 'dev_cut_index_list.json', 'w') as f:
    json.dump(dev_cut_index_dict, f, ensure_ascii=False)


train_dict = {'text': train_cut_text_list, 'unknownEntities': train_cut_label_list}
train_df = pd.DataFrame(train_dict)

dev_dict = {'text': dev_cut_text_list, 'unknownEntities': dev_cut_label_list}
dev_df = pd.DataFrame(dev_dict)

test_dict = {'text': test_cut_text_list}
test_df = pd.DataFrame(test_dict)

print('训练集:', train_df.shape)
print('验证集:', dev_df.shape)
print('测试集:', test_df.shape)

# 构造训练集、验证集与测试集
with codecs.open(data_dir + 'train.txt', 'w', encoding='utf-8') as up:
    for row in train_df.iloc[:].itertuples():
        # print(row.unknownEntities)

        text_lbl = row.text
        entitys = str(row.unknownEntities).split(';')
        for entity in entitys:
            text_lbl = text_lbl.replace(entity, 'Ё' + (len(entity) - 1) * 'Ж')

        for c1, c2 in zip(row.text, text_lbl):
            if c2 == 'Ё':
                up.write('{0} {1}\n'.format(c1, 'B-ORG'))
            elif c2 == 'Ж':
                up.write('{0} {1}\n'.format(c1, 'I-ORG'))
            else:
                up.write('{0} {1}\n'.format(c1, 'O'))
        up.write('\n')

with codecs.open(data_dir + 'dev.txt', 'w', encoding='utf-8') as up:
    for row in dev_df.iloc[:].itertuples():
        # print(row.unknownEntities)
        text_lbl = row.text
        entitys = str(row.unknownEntities).split(';')
        for entity in entitys:
            text_lbl = text_lbl.replace(entity, 'Ё' + (len(entity) - 1) * 'Ж')

        for c1, c2 in zip(row.text, text_lbl):
            if c2 == 'Ё':
                up.write('{0} {1}\n'.format(c1, 'B-ORG'))
            elif c2 == 'Ж':
                up.write('{0} {1}\n'.format(c1, 'I-ORG'))
            else:
                up.write('{0} {1}\n'.format(c1, 'O'))

        up.write('\n')

with codecs.open(data_dir + 'test.txt', 'w', encoding='utf-8') as up:
    for row in test_df.iloc[:].itertuples():

        text_lbl = row.text
        for c1 in text_lbl:
            up.write('{0} {1}\n'.format(c1, 'O'))

        up.write('\n')
