# Poetry-generate
三种方法：n-gram、lstm、tansformer实现古诗词生成

## n-gram方法
该代码只是简单的实现，其中还有和多地方可以优化。比如：
       * 1.训练数据的每首诗词长度不统一，如果对诗的要求更高的话，可以将训练数据全部换成五言绝句等长度、格式较统一的诗词。
       * 2.在预测方面：  缺点：首先这里直接采用随机预测可能的结果，这样预测更加灵活，但不够精准
                         改进：可以将list改为dict来存储预测的词语，通过dict将预测词语的频率记录下来，方便后面预测时，可以选择贪婪算法、按概率预测等更加好的方法
       * 3. 另外如果训练数据库很大，建议将n改得更大些，提高预测准确度
## LSTM
使用.py文件即可，该文件藏头诗尚可，然而不能很好的学习标点符号，可能跟上下依赖关系即位置关系不能很好学习有关
## Transformer
请点击下列超联集
[GitHub](https://github.com)
