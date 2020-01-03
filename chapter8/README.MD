## 2019BDCI互联网金融新实体发现（单模成绩第一，最终成绩第二）
* 实践是检验理论的唯一标准。为此，笔者将通过中国计算机学会举办的[2019CCF大数据与计算智能大赛的互联网金融新实体发现竞赛](https://www.datafountain.cn/competitions/361)作为实践，让读者们在了解预训练模型强大的同时，顺便掌握打比赛的流程。
* 笔者的代码在竞赛中获得了全国第二的成绩，加之笔者的单模成绩在TOP5中最好以及代码的复用性与解耦性强，值得大家学习与借鉴。

### 赛题分析
  * 赛题要求从提供的金融网络文本中识别出现的未知金融实体，包括金融平台名、企业名、项目名。网络文本包括标题和内容，标题和内容至少有一个不为空。
  
  * 数据集中的文本长度分布下图所示，文本长度0-500的数据有3615条，超过500的则有6390条。大部分数据文本长度较长。其中文本最短长度为4，最大长度为32787，平均长度为1311。
  * 在训练集中还存在200多条数据有标签谬误。
  * 数据集中出现了部分噪音，包括一些HTML文字和特殊字符。
  * 可以看出，数据集存在文本过长，噪声过多等问题。
![image]( https://github.com/ChileWang0228/Deep-Learning-With-Python/blob/master/chapter8/images/%E6%96%87%E6%9C%AC%E9%95%BF%E5%BA%A6%E7%BB%9F%E8%AE%A1.png
)
 
### 数据处理
  * 针对数据分析中提到的噪音及标签谬误等问题，使用了正则表达式定位噪声与标签谬误数据，进而清洗噪声与修正标签谬误。
  * 针对数据分析中提到的文本过长的问题，本团队对金融文本采用了按句子切割的方法，以标点符号优先级对句子进行切割，并按原顺序进行重组，当重组的句子长度超过512时，则新生成一条子数据并对剩余句子重复执行上述过程，直到所有的句子都被组装完成。这种数据处理的方式有效的解决了数据集文本长度过长的问题，并且完整地利用了数据信息。
  

### 模型方法
  * BERT + BILSTM + CRF
  * BERT + IDCNN + CRF
  * 动态权重BERT + BILSTM + CRF
  * 动态权重BERT + IDCNN + CRF
![image](https://github.com/ChileWang0228/Deep-Learning-With-Python/blob/master/chapter8/images/BILSTM.png)
![image](https://github.com/ChileWang0228/Deep-Learning-With-Python/blob/master/chapter8/images/IDCNN.png)
![image](https://github.com/ChileWang0228/Deep-Learning-With-Python/blob/master/chapter8/images/%E5%8A%A8%E6%80%81%E6%9D%83%E9%87%8D%E8%9E%8D%E5%90%88.png)


### 模型融合
  * 上面提到的四种异构单模分别搭载其他预训练模型{BERT_WWM, ROBERTA, ERNIE}得到多种异构单模。
  * 将recall>0.64的异构单模抽出来进行预测，得到每个异构单模的文字结果。
  * 设置阈值 thresh=ensemble_model_num / 3， 当这个实体在thresh个以上的模型出现时，则保留。
  ![image](https://github.com/ChileWang0228/Deep-Learning-With-Python/blob/master/chapter8/images/%E6%A8%A1%E5%9E%8B%E8%9E%8D%E5%90%88.png)
  
### 后处理规则
  * 对含有标点符号的文字结果进行了一些一般化后处理工作。
```python
"""
场景1：extra_chars = set("!,:@_！，：。[]")  # 直接干掉。
场景2：extra_chars = set("·")  # 在头or尾直接舍弃
场景3：extra_chars = set(".")  # 去头去尾，‘.’在中间，要保证前后都是英文字符，出现中文字符则直接舍弃。
场景4：extra_chars = set("#$￥%+<=>?^`{|}~#%？《{}“”‘’【】")  # 直接替换成''
场景5：extra_chars = set("-&\/&")  # 去头去尾 ‘-’在实体中间是合法实体
场景6：extra_chars = set("()（）")   # 若不是对应匹配的括号，括号半边在头与尾，替换成‘’，括号在实体中间则舍弃
场景7：extra_chars = set('、；、')  # split
场景8：删除单字情况和删除训练集存在的实体
"""
```
  * 删除训练集出现过的实体（赛题规则）。
  
