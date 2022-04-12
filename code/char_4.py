import os
import math
import pickle
import random
import time
import sys

def init(): # 初始化字频，条件频率矩阵，拼音汉字对照表,并将汉字与数字id一一对应起来
    StarTime = time.time()
    Pinyin2Char = {}  # 汉字拼音对照表  例如：Pinyin2Char['qing']={'1','4','9',....}
    with open('../训练数据/汉字表与对照表/拼音汉字对照表.txt', encoding='utf-8') as f: #初始化汉字拼音汉字对照表
        # 考虑一个汉字可以有多个拼音
        for line in f.readlines():  # 例如: line = 'an 安 按 案 ... 暗\n'
            line = line[:-1]  # 去掉换行
            chars = line.split(' ')  # 同音汉字列表
            key = chars[0]
            del chars[0]  # 第一个元素是拼音
            Pinyin2Char[key] = chars # 加入拼音汉字对照表
    pickle.dump(Pinyin2Char, open('../模型/字四元/拼音汉字对照表.pkl', 'wb'))  # 保存拼音汉字对照表
    EndTime = time.time()
    print("init函数运行时间为:", EndTime - StarTime, "s")

def count(): # 统计字频,两字相邻频率,三字相邻频率
    StartTime = time.time()
    SingleFreq ={}
    DoubleFreq ={}
    TripleFreq ={}
    QuartFreq = {}
    CharListFile = open('../训练数据/汉字表与对照表/一二级汉字表.txt', encoding='utf-8')
    CharList = CharListFile.readlines()[0]
    for i in range(len(CharList)):
        SingleFreq[CharList[i]]=0
    DirName='../训练数据/训练语料/'
    FileList = os.listdir(DirName) # 列出语料库所有文件
    CountTxt = 0 # 训练到第几个文件了
    for item in FileList: # 遍历语料库文件
        CountTxt += 1
        if CountTxt>=2:
            break
        CountLine = 0 #训练到这个文件的第几行了
        f=open(DirName + item, encoding='utf-8')
        TxtLines = f.readlines() #把这个文件的内容按行读取
        TotalLines = len(TxtLines) #这个文件一共有多少行
        for line in TxtLines:
            CountLine += 1
            if CountLine % 1000 == 0: # 每1000行输出一下进度
                print("现在训练到第", CountTxt, "/",len(FileList),"个文件的第", CountLine, "/", TotalLines, "行",
                      "四元字文件大小为：",sys.getsizeof(QuartFreq),"B")
                # print(TripleFreq)
            sentences = [] # 这一行里的很多句子
            sentence = '' #某一个句子
            for c in line:
                if c in SingleFreq.keys(): # 如果是汉字，就加入临时的句子 # 如果是汉字但未在字典中也不用管
                    sentence+=c
                else:
                    if len(sentence)>=4:
                        sentences.append(sentence)
                    sentence=''
            for sentence in sentences: # 对每个短句中的汉字进行遍历
                l=len(sentence) # 这个句子的长度
                for i in range(l): # 这里用i是因为需要方便地知道上一个字符是什么
                    SingleFreq[sentence[i]] += 1
                    # 计算两字相邻频率(条件频率)，AdjoinFreq['清']['华']=50 表示在'清'出现的情况下'华'出现的频率为50
                    if i >= 1: # 句子的最后一个汉字不用算相邻频率
                        if sentence[i] in DoubleFreq.keys():
                            if sentence[i-1] in DoubleFreq[sentence[i]].keys():
                                DoubleFreq[sentence[i]][sentence[i-1]] += 1
                            else:
                                DoubleFreq[sentence[i]][sentence[i-1]]=1
                        else:
                            DoubleFreq[sentence[i]] = {}
                            DoubleFreq[sentence[i]][sentence[i - 1]] = 1
                    if i >= 2:
                        if sentence[i] in TripleFreq.keys():
                            if sentence[i - 1] in TripleFreq[sentence[i]].keys():
                                if sentence[i - 2] in TripleFreq[sentence[i]][sentence[i - 1]].keys():
                                    TripleFreq[sentence[i]][sentence[i - 1]][sentence[i - 2]] += 1
                                else:
                                    TripleFreq[sentence[i]][sentence[i - 1]][sentence[i - 2]]=1
                            else:
                                TripleFreq[sentence[i]][sentence[i - 1]]={}
                                TripleFreq[sentence[i]][sentence[i - 1]][sentence[i - 2]]=1
                        else:
                            TripleFreq[sentence[i]]={}
                            TripleFreq[sentence[i]][sentence[i - 1]]={}
                            TripleFreq[sentence[i]][sentence[i - 1]][sentence[i - 2]] = 1
                    if i >= 3:
                        if sentence[i] in QuartFreq.keys():
                            if sentence[i - 1] in QuartFreq[sentence[i]].keys():
                                if sentence[i - 2] in QuartFreq[sentence[i]][sentence[i - 1]].keys():
                                    if sentence[i - 3] in QuartFreq[sentence[i]][sentence[i - 1]][sentence[i - 2]].keys():
                                        QuartFreq[sentence[i]][sentence[i - 1]][sentence[i - 2]][sentence[i - 3]] += 1
                                    else:
                                        QuartFreq[sentence[i]][sentence[i - 1]][sentence[i - 2]][sentence[i - 3]]=1
                                else:
                                    QuartFreq[sentence[i]][sentence[i - 1]][sentence[i - 2]] = {}
                                    QuartFreq[sentence[i]][sentence[i - 1]][sentence[i - 2]][sentence[i - 3]]=1
                            else:
                                QuartFreq[sentence[i]][sentence[i - 1]]={}
                                QuartFreq[sentence[i]][sentence[i - 1]][sentence[i - 2]] = {}
                                QuartFreq[sentence[i]][sentence[i - 1]][sentence[i - 2]][sentence[i - 3]] = 1
                        else:
                            QuartFreq[sentence[i]]={}
                            QuartFreq[sentence[i]][sentence[i - 1]] = {}
                            QuartFreq[sentence[i]][sentence[i - 1]][sentence[i - 2]] = {}
                            QuartFreq[sentence[i]][sentence[i - 1]][sentence[i - 2]][sentence[i - 3]] = 1
        print(SingleFreq)
        print("--------------------------------------------------")
        print(DoubleFreq['清'])
        print("--------------------------------------------------")
        print(TripleFreq['华']['清'])
        print("--------------------------------------------------")
        print(QuartFreq['华']['清'])
    pickle.dump(SingleFreq, open('../模型/字四元/1.4/字频.pkl', 'wb')) # 保存字频，下次启动就不用再算了
    pickle.dump(DoubleFreq, open('../模型/字四元/1.4/两字相邻频率.pkl', 'wb'))  # 保存条件频率矩阵
    pickle.dump(TripleFreq, open('../模型/字四元/1.4/三字相邻频率.pkl', 'wb'))  # 保存条件频率矩阵
    pickle.dump(QuartFreq, open('../模型/字四元/1.4/四字相邻频率.pkl', 'wb'))  # 保存条件频率矩阵
    EndTime = time.time()
    print("count函数运行时间为:",  EndTime - StartTime, "s")

def viterbi(PinSentence, Pinyin2Char,SingleFreq, DoubleFreq ,TripleFreq, QuartFreq):
    Pin2Han = [] # Pin2Han的第一维为列表，第二维为字典，即Pin2Han是一个字典列表，例如：Pin2Han[2]['清']=0.005(为当前算得的句子概率)
    LastChar=[] #存储当前状态是从上一层的哪个状态过来的，便于以链表的形式吧结果输出出来
    gamma=50 # gamma参数，对字频小于gamma的字停用,主要作用是剪枝，对准确率的影响不大
    for i in range(len(PinSentence)):
        Pin2Han.append({})
        LastChar.append({})
        for c in Pinyin2Char[PinSentence[i]]:
            if SingleFreq[c] > gamma:
                Pin2Han[i][c] = -10000
    l=len(PinSentence)
    epsilon=1e-20
    alpha=3e-2 #平滑操作的参数，用alpha * P(w_i|w_{i-1}) + (1-alpha) * P(w_i)代替P(w_i|w_{i-1})
    beta=1e-10
    for i in range(l):
        if i == 0: # 第一个字对应的拼音
            for c in Pin2Han[i].keys(): # 遍历这个拼音对应的所有字
                Pin2Han[i][c] = math.log(SingleFreq[c]) # 第一个字，用这个字出现的概率作为条件概率
        elif i == 1: # 第一个字对应的拼音
            for c in Pin2Han[i].keys(): # 遍历这个拼音对应的所有字
                for PrevChar in Pin2Han[i-1].keys():# 不用Pinyin2Char[PinSentence[i-1]],考虑到上一层可能剪枝
                    temp = -10000
                    if c in DoubleFreq.keys():
                        if PrevChar in DoubleFreq[c].keys():
                            x=DoubleFreq[c][PrevChar]
                            temp = Pin2Han[i - 1][PrevChar] + math.log(4000000 * x / SingleFreq[PrevChar] + 0.002* SingleFreq[c])
                        else:
                            temp = Pin2Han[i - 1][PrevChar] + math.log(0.001 * SingleFreq[c])
                    else:
                        temp = Pin2Han[i - 1][PrevChar] + math.log(0.000001 * SingleFreq[c])
                    if temp> Pin2Han[i][c]:
                        Pin2Han[i][c]=temp
                        LastChar[i][c]=PrevChar
        elif i==2: # 双重循环，第一重是这一层的所有汉字，第二层是上一层的所有汉字
            for c in Pin2Han[i].keys():
                for PrevChar in Pin2Han[i-1].keys():# 不用Pinyin2Char[PinSentence[i-1]],考虑到上一层可能剪枝
                    x = 0
                    y = 0
                    z = 0
                    # 以清华大学为例
                    PrevPrevChar = LastChar[i - 1][PrevChar]  # 前面的前面的一个字符
                    if PrevChar in DoubleFreq.keys():
                        if PrevPrevChar in DoubleFreq[PrevChar]:
                            x=DoubleFreq[PrevChar][PrevPrevChar] # 华大的频率
                    if c in DoubleFreq.keys():
                        if PrevChar in DoubleFreq[c]:
                            y=DoubleFreq[c][PrevChar] # 大学的频率
                    if c in TripleFreq.keys():
                        if PrevChar in TripleFreq[c].keys():
                            if PrevPrevChar in TripleFreq[c][PrevChar].keys():
                                z=TripleFreq[c][PrevChar][PrevPrevChar] # 华大学的频率
                                temp = Pin2Han[i - 1][PrevChar] + math.log(9000000* z / x + 2000000 * y/SingleFreq[PrevChar] + 0.000001*SingleFreq[c])
                            else:
                                temp = Pin2Han[i - 1][PrevChar] + math.log(2000000 * y/SingleFreq[PrevChar] + 0.000001 * SingleFreq[c])
                        else:
                            temp = Pin2Han[i - 1][PrevChar] + math.log(0.000001 * SingleFreq[c])
                    else:
                        temp = Pin2Han[i - 1][PrevChar] + math.log(0.00000001 * SingleFreq[c])
                    if temp> Pin2Han[i][c]:
                        Pin2Han[i][c]=temp
                        LastChar[i][c]=PrevChar
        else:
            for c in Pin2Han[i].keys():
                for PrevChar in Pin2Han[i-1].keys():# 不用Pinyin2Char[PinSentence[i-1]],考虑到上一层可能剪枝
                    x = 0
                    y = 0
                    z = 0
                    w = 0
                    temp=-10000
                    # 例：清华大学
                    PrevPrevChar = LastChar[i - 1][PrevChar]  # 前面的前面的一个字符
                    PrevPrevPrevChar = LastChar[i - 2][PrevPrevChar]  # 前面的前面的前面的一个字符
                    if PrevChar in DoubleFreq.keys():
                        if PrevPrevChar in DoubleFreq[PrevChar]:
                            x=DoubleFreq[PrevChar][PrevPrevChar] #华大的频率
                    if c in DoubleFreq.keys():
                        if PrevChar in DoubleFreq[c]:
                            y=DoubleFreq[c][PrevChar] # 大学的频率
                    if c in TripleFreq.keys():
                        if PrevChar in TripleFreq[c].keys():
                            if PrevPrevChar in TripleFreq[c][PrevChar].keys():
                                z = TripleFreq[c][PrevChar][PrevPrevChar] # 华大学的频率
                                temp = Pin2Han[i - 1][PrevChar] + math.log(9000000 * z/x +2000000*y/SingleFreq[PrevChar]+ 0.000003 * SingleFreq[c])
                            else:
                                temp = Pin2Han[i - 1][PrevChar] + math.log(2000000 * y/SingleFreq[PrevChar] + 0.0000001 * SingleFreq[c])
                        else:
                            temp = Pin2Han[i - 1][PrevChar] + math.log(0.000001 * SingleFreq[c])
                    else:
                        temp = Pin2Han[i - 1][PrevChar] + math.log(0.00000001 * SingleFreq[c])
                    if PrevChar in TripleFreq.keys():
                        if PrevPrevChar in TripleFreq[PrevChar].keys():
                            if PrevPrevPrevChar in TripleFreq[PrevChar][PrevPrevChar].keys():
                                w = TripleFreq[PrevChar][PrevPrevChar][PrevPrevPrevChar] # 清华大的频率
                    if c in QuartFreq.keys():
                        if PrevChar in QuartFreq[c].keys(): #前面的一个字符
                            if PrevPrevChar in QuartFreq[c][PrevChar].keys():
                                if PrevPrevPrevChar in QuartFreq[c][PrevChar][PrevPrevChar].keys():
                                    u = QuartFreq[c][PrevChar][PrevPrevChar][PrevPrevPrevChar]
                                    temp = Pin2Han[i - 1][PrevChar] + math.log(0* u / w + 9000000 * z/x +2000000*y/SingleFreq[PrevChar]+ 0.000003*SingleFreq[c])
                    #            else:
                    #                temp = Pin2Han[i - 1][PrevChar] + math.log(9000000 * z/x +2000000*y/SingleFreq[PrevChar]+ 0.000003 * SingleFreq[c])
                    #        else:
                    #            temp = Pin2Han[i - 1][PrevChar] + math.log(2000000 * y/SingleFreq[PrevChar] + 0.0000001 * SingleFreq[c])
                    #    else:
                    #        temp = Pin2Han[i - 1][PrevChar] + math.log(0.000001 * SingleFreq[c])
                    #else:
                    #    temp = Pin2Han[i - 1][PrevChar] + math.log(0.00000001 * SingleFreq[c])
                    if temp> Pin2Han[i][c]:
                        Pin2Han[i][c]=temp
                        LastChar[i][c]=PrevChar
    EndChar=''
    Max=-10000
    for c in Pin2Han[l - 1].keys():
        if Pin2Han[l - 1][c] >Max:
            Max=Pin2Han[l - 1][c]
            EndChar=c
    CharList=[]
    CharList.append(EndChar)
    for i in range(l-1):
        EndChar=LastChar[l-1-i][EndChar]
        CharList.append(EndChar)
    l=len(CharList)
    HanSentence=''
    for i in range(l):
        HanSentence+=CharList[l-1-i]
    # print(HanSentence)
    return HanSentence

def predict():
    StartTime = time.time()
    print("start")
    InFile = "test/IN-3.txt"#sys.argv[1] #"test/IN-1.txt"   命令行运行(输入两个参数，分别为输入和输出txt文件名)：python 字二元.py input.txt output.txt
    OutFile = "test/OUT-3.6.txt"#sys.argv[2]  #"test/OUT-1.1.txt"  默认输出到data文件夹下
    Pinyin2Char = pickle.load(open('../模型/字四元/拼音汉字对照表.pkl', 'rb'))  # 载入拼音汉字对照表
    SingleFreq = pickle.load(open('../模型/字四元/1.5/字频.pkl', 'rb'))  # 载入字频
    DoubleFreq = pickle.load(open('../模型/字四元/1.5/两字相邻频率.pkl', 'rb'))  # 载入条件频率矩阵
    TripleFreq = pickle.load(open('../模型/字四元/1.5/三字相邻频率.pkl', 'rb'))  # 载入条件频率矩阵
    QuartFreq = pickle.load(open('../模型/字四元/1.5/四字相邻频率.pkl', 'rb'))  # 载入条件频率矩阵
    # sum1=sum(SingleFreq.values())
    # sum2=0
    # for i in DoubleFreq.keys():
    #     sum2+=sum(DoubleFreq[i].values())
    # sum3=0
    # for i in TripleFreq.keys():
    #     for j in TripleFreq[i].keys():
    #         sum3+=sum(TripleFreq[i][j].values())
    # print(sum1,sum2,sum3)
    #加载上面的大文件会占用大量时间（超过90%）
    CountLine = 0 # 现在预测到第几行了
    IN=open('../data/' + InFile, encoding='utf-8') # 输入文件
    OUT = open('../data/' + OutFile, 'w',encoding="utf-8")
    # 读取每一行拼音
    for line in IN.readlines():
        CountLine += 1
        PinSentence = line[:-1].split(' ') # 将字符串转成列表
        HanSentence=viterbi(PinSentence, Pinyin2Char, SingleFreq, DoubleFreq, TripleFreq, QuartFreq) # 使用viterbi算法得到预测的句子
        OUT.write(HanSentence+'\n') # 保存结果
        # print("预测到第", CountLine, "行")
        if(CountLine%100==0):
            print("预测到第", CountLine, "行")
    EndTime = time.time()
    print("predict函数运行时间为:", (EndTime - StartTime), "s")

def CreateTestData(): #构造测试集
    IN=open('../data/test/测试集语料（未清洗）.txt', encoding='utf-8')  # 输入文件
    CharFreq = pickle.load(open('../模型/字二元/版本1/字频.pkl', 'rb'))  # 加载字频字典 例如：CharFreq['清']=100
    TxtLines = IN.readlines()  # 把这个文件的内容按行读取
    TotalLines = len(TxtLines)  # 这个文件一共有多少行
    CountLine = 0  # 读取到这个文件的第几行了
    sentences = []  # 这一个文件里的很多句子
    for line in TxtLines:
        CountLine += 1
        if CountLine % 1000 == 0:  # 每1000行输出一下进度
            print("现在读取到第", CountLine, "/", TotalLines, "行")
        sentence = ''  # 某一个句子
        for c in line:
            if c in CharFreq.keys():  # 如果是汉字，就加入临时的句子 # 如果是汉字但未在字典中也不用管
                sentence += c
            else:
                if len(sentence) > 4:
                    sentences.append(sentence)
                sentence = ''
    NumList=[] # 这里使用构造的NumList来提高效率，因为shuffle的实现是值交换，如果交换应该会很慢
    l = len(sentences)
    for i in range(l):
        NumList.append(i)
    random.shuffle(NumList)
    OUT = open('../data/test/测试集句子（清洗后）.txt', 'w')
    for i in range(l):
        OUT.write(sentences[NumList[i]] + '\n')

def check(): # 计算模型的准确度，包括字准确度和句准确度
    OUT = open('../data/test/OUT-3.6.txt',encoding="utf-8")
    Answer = open('../data/test/Answer-3.txt',encoding="utf-8")
    OutLines=OUT.readlines()
    AnswerLines = Answer.readlines()
    l=len(OutLines) #总的句子数
    RightLines=0
    Chars=0
    RightChars=0
    for i in range(l):
        if OutLines[i]==AnswerLines[i]:
            RightLines+=1
        s=len(OutLines[i])-1 #这一句一共有多少个字，去掉换行。
        Chars+=s
        for j in range(s):
            if OutLines[i][j]==AnswerLines[i][j]:
                RightChars+=1
    print("句准确率为：",round(100*RightLines/l,3),"%")
    print("字准确率为：", round(100*RightChars / Chars, 3),"%")


if __name__ == '__main__':
    # init() # 初始化字频，条件频率矩阵，拼音汉字对照表
    # count()  # 统计并保存语料库中的字频和两字相邻次数
    predict()  # 对输入的数据进行预测并将结果保存下来
    # CreateTestData() # 构造测试集
    check() # 检验模型的准确率

