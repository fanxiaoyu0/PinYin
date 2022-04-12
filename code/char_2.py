import os
import math
import pickle
import random
import time
from tqdm import tqdm
random.seed(1024)

def init(): # 初始化字频，两字相邻频率矩阵，拼音汉字对照表
    StarTime = time.time()
    CharFreq = {}  # 字频字典 例如：CharFreq['清']=100
    AdjoinFreq = {}  # 两字相邻频率字典，例如：AdjoinFreq['清']['华']=50
    Pinyin2Char = {}  # 汉字拼音对照表  例如：Pinyin2Char['qing']={'清','请','情',....}
    with open('../data/汉字表与对照表/拼音汉字表.txt', encoding='utf-8') as f:
        # 初始化两个字典（只考虑拼音汉字表中的汉字即可）
        for line in f.readlines():  # 例如: line = 'an 安 按 案 ... 暗\n'
            line = line[:-1]  # 去掉换行
            chars = line.split(' ')  # 同音汉字列表
            key = chars[0]
            del chars[0]  # 第一个元素是拼音
            Pinyin2Char[key] = chars  # 加入拼音汉字对照表
            # 将字频字典的每个字出现的频率初始化成0
            for c in chars:
                CharFreq[c] = 0
    # 将两字相邻频率矩阵的每个元素初始化成0
    chars = CharFreq.keys()
    for c in chars:
        AdjoinFreq[c] = {}
        for C in chars:
            AdjoinFreq[c][C] = 0
    # print(Pinyin2Char)
    pickle.dump(Pinyin2Char, open('../model/char_2/拼音汉字对照表.pkl', 'wb'))  # 保存拼音汉字对照表
    pickle.dump(CharFreq, open('../model/char_2/字频.pkl', 'wb'))  # 保存字频
    pickle.dump(AdjoinFreq, open('../model/char_2/两字相邻频率.pkl', 'wb')) 
    EndTime = time.time()
    print("init函数运行时间为:", EndTime - StarTime, "s")

def get_sentences(): # 将原始数据整理为干净的句子列表
    StartTime = time.time()
    CharFreq = pickle.load(open('../model/char_2/字频.pkl', 'rb'))  # 加载字频字典 例如：CharFreq['清']=100
    FileList = os.listdir('../data/sina_news_gbk/') # 列出语料库所有文件
    for CountTxt,FileName in enumerate(FileList): # 遍历语料库文件
        with open('../data/sina_news_gbk/' + FileName, encoding='gbk') as f:
            sentences = [] # 这一行里的很多句子
            TxtLines = f.readlines() #把这个文件的内容按行读取
            CountLine = 0 #训练到这个文件的第几行了
            for line in tqdm(TxtLines,desc="统计到第"+str(CountTxt)+"个文件"):
                CountLine += 1
                # if CountLine % 1000 == 0: # 每1000行输出一下进度
                #     print("现在训练到第", CountTxt, "/",len(FileList),"个文件的第", CountLine, "/", len(TxtLines), "行")
                    # break
                sentence = '' #某一个句子
                for c in line:
                    if c in CharFreq.keys(): # 如果是汉字，就加入临时的句子 # 如果是汉字但未在字典中也不用管
                        sentence+=c
                    else:
                        # 遇到非汉字就截断，认为上一句话结束，将上一句话加入句子列表中，忽略只有一个字的句子
                        if len(sentence)>1 and sentence!="原标题":
                            sentences.append(sentence)
                            # if random.random()<1e-6:
                                # print(sentence)
                        sentence=''
            pickle.dump(sentences, open('../data/句子列表/'+FileName[:-4]+'.pkl', 'wb'))
            print(len(sentences))
        # break
    # print(sentences)
    # print(len(sentences))
    # fdskfj
    
    EndTime = time.time()
    print("get_sentences函数运行时间为:",  EndTime - StartTime, "s")

# ，并划分为训练集、验证集、测试集

def create_dataset():
    StartTime = time.time()
    validSentenceList=[]
    testSentenceList=[]
    for index,FileName in enumerate(os.listdir('../data/句子列表/')):
        sentences=pickle.load(open('../data/句子列表/' + FileName, 'rb'))
        validSentenceNumber=len(sentences)//5000
        testSentenceNumber=len(sentences)//5000
        for i in tqdm(range(validSentenceNumber),desc=str(index)+":valid"):
            temp=random.randint(0,len(sentences)-1)
            validSentenceList.append(sentences[temp])
            sentences.pop(temp)
        for i in range(testSentenceNumber):
            temp=random.randint(0,len(sentences)-1)
            testSentenceList.append(sentences[temp])
            sentences.pop(temp)
        pickle.dump(sentences, open('../data/train/'+FileName[:-4], 'wb'))
    with open('../data/valid/汉字句子.txt', 'w', encoding='utf-8') as f:
        # random.shuffle(validSentenceList)
        for sentence in validSentenceList:
            f.write(sentence+'\n')
    with open('../data/test/汉字句子.txt', 'w', encoding='utf-8') as f:
        for sentence in testSentenceList:
            f.write(sentence+'\n')
    EndTime = time.time()
    print("get_sentences函数运行时间为:",  EndTime - StartTime, "s")


def count(): # 统计字频和两字相邻频率
    StartTime = time.time()
    CharFreq = pickle.load(open('../model/char_2/字频.pkl', 'rb'))  # 加载字频字典 例如：CharFreq['清']=100
    AdjoinFreq = pickle.load(open('../model/char_2/两字相邻频率.pkl', 'rb'))  # 加载两字相邻频率字典，例如：AdjoinFreq['清']['华']=50
    DirName='../data/sina_news_gbk/'
    FileList = os.listdir(DirName) # 列出语料库所有文件
    CountTxt = 0 # 统计到第几个文件了
    for item in FileList: # 遍历语料库文件
        CountTxt += 1
        CountLine = 0 #训练到这个文件的第几行了
        f=open(DirName + item, encoding='gbk')
        TxtLines = f.readlines() #把这个文件的内容按行读取
        TotalLines = len(TxtLines) #这个文件一共有多少行
        for line in TxtLines:
            CountLine += 1
            if CountLine % 1000 == 0: # 每1000行输出一下进度
                print("现在训练到第", CountTxt, "/",len(FileList),"个文件的第", CountLine, "/", TotalLines, "行")
                break
            sentences = [] # 这一行里的很多句子
            sentence = '' #某一个句子
            for c in line:
                if c in CharFreq.keys(): # 如果是汉字，就加入临时的句子 # 如果是汉字但未在字典中也不用管
                    sentence+=c
                else:
                    # 遇到非汉字就截断，认为上一句话结束，将上一句话加入句子列表中
                    if len(sentence)>1:
                        sentences.append(sentence)
                    sentence=''
            for sentence in sentences: # 对每个短句进行遍历
                l=len(sentence) # 这个句子的长度
                for i in range(l): # 这里用i是因为需要方便地知道上一个字符是什么
                    CharFreq[sentence[i]] += 1
                    # 计算两字相邻频率(条件频率)，AdjoinFreq['清']['华']=50 表示在'清'出现的情况下'华'出现的频率为50
                    if i < l - 1: # 句子的最后一个汉字不用算相邻频率
                        AdjoinFreq[sentence[i]][sentence[i + 1]] += 1
        f.close()
        break
    print(CharFreq)
    print(AdjoinFreq['清'])
    pickle.dump(CharFreq, open('../model/char_2/字频.pkl', 'wb')) # 保存字频，下次启动就不用再算了
    pickle.dump(AdjoinFreq, open('../model/char_2/两字相邻频率.pkl', 'wb'))  # 保存两字相邻频率矩阵
    EndTime = time.time()
    print("count函数运行时间为:",  EndTime - StartTime, "s")

def viterbi(PinSentence, Pinyin2Char,CharFreq, AdjoinFreq ):
    Pin2Char = [] # Pin2Char的第一维为列表，第二维为字典，即Pin2Char是一个字典列表，例如：Pin2Char[2]['清']=0.005(为当前算得的句子概率)
    LastChar=[] # 存储当前状态是从上一层的哪个状态过来的，便于以链表的形式把结果输出出来
    gamma=10 # gamma参数，对字频小于gamma的字停用,主要作用是剪枝，对准确率的影响不大
    for i in range(len(PinSentence)):
        Pin2Char.append({})
        LastChar.append({})
        for c in Pinyin2Char[PinSentence[i]]:
            if CharFreq[c] > gamma:
                Pin2Char[i][c] = -10000
    l=len(PinSentence)
    alpha=1e-10 #平滑操作的参数，用alpha * P(w_i|w_{i-1}) + (1-alpha) * P(w_i)代替P(w_i|w_{i-1})
    for i in range(l):
        if i == 0: # 第一个字对应的拼音
            for c in Pin2Char[i].keys(): # 遍历这个拼音对应的所有字
                Pin2Char[i][c] = math.log(CharFreq[c]) # 第一个字，用这个字出现的概率作为条件概率
        else: # 双重循环，第一重是这一层的所有汉字，第二层是上一层的所有汉字
            for c in Pin2Char[i].keys():
                for PrevChar in Pin2Char[i-1].keys():# 不用Pinyin2Char[PinSentence[i-1]],考虑到上一层可能剪枝
                    temp=-10000
                    if AdjoinFreq[PrevChar][c]>0:
                        temp = Pin2Char[i - 1][PrevChar] + math.log(1000000000*AdjoinFreq[PrevChar][c]/CharFreq[PrevChar]+0.0001*CharFreq[c])
                    else:
                        temp = Pin2Char[i - 1][PrevChar] + math.log(0.0001 * CharFreq[c])
                    if temp> Pin2Char[i][c]:
                        Pin2Char[i][c]=temp
                        LastChar[i][c]=PrevChar
    EndChar=''
    Max=-10000
    for c in Pin2Char[l - 1].keys():
        if Pin2Char[l - 1][c] >Max:
            Max=Pin2Char[l - 1][c]
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
    InFile = "test/IN-3.txt"#sys.argv[1] #"test/IN-1.txt"   命令行运行(输入两个参数，分别为输入和输出txt文件名)：python 字二元.py input.txt output.txt
    OutFile = "test/OUT-3.2.txt"#sys.argv[2]  #"test/OUT-1.1.txt"  默认输出到data文件夹下
    Pinyin2Char = pickle.load(open('../模型/字二元/拼音汉字对照表.pkl', 'rb'))  # 载入拼音汉字对照表
    CharFreq = pickle.load(open('../模型/字二元/1.1/字频.pkl', 'rb'))  # 载入字频
    AdjoinFreq = pickle.load(open('../模型/字二元/1.1/条件频率矩阵.pkl', 'rb'))  # 载入条件频率矩阵
    #加载上面的大文件会占用大量时间（超过90%）
    CountLine = 0 # 现在预测到第几行了
    IN=open('../data/' + InFile, encoding='utf-8') # 输入文件
    OUT = open('../data/' + OutFile, 'w',encoding="utf-8")

    # 读取每一行拼音
    for line in IN.readlines():
        CountLine += 1
        PinSentence = line[:-1].split(' ') # 将字符串转成列表
        HanSentence=viterbi(PinSentence, Pinyin2Char, CharFreq, AdjoinFreq) # 使用viterbi算法得到预测的句子
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
    OUT = open('../data/test/OUT-3.2.txt',encoding="utf-8")
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
    # init() # 初始化字频，两字相邻频率矩阵，将拼音汉字对照表转成pkl文件
    # get_sentences()
    create_dataset()
    # count()  # 统计并保存语料库中的字频和两字相邻次数
    # predict()  # 对输入的数据进行预测并将结果保存下来
    # CreateTestData() # 构造测试集
    # check() # 检验模型的准确率
    print("All is well!")

