from functools import reduce
class Perceptron(object):
    def __init__(self, input_num, activator):
        '''
        input_num输入参数个数
        activator激活函数
        '''
        self.activator = activator
        # 将每个参数的权重初始化为0
        self.weights = [0.0 for _ in range(input_num)]
        # 偏置项初始化为0
        self.bias = 0.0
    
    def __str__(self):
        '''
        打印学习到的权重
        '''
        return "weights\t:%s\nbias\t:%f\n" % (self.weights, self.bias)
    
    def predict(self, input_vec):
        '''
        输入向量，输出感知器的计算结果
        '''
        # 把input_vec[x1,x2,x3...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用map函数计算[x1*w1, x2*w2, x3*w3]
        # 最后利用reduce求和
        return self.activator(
            reduce(lambda a, b: a + b,
                   list(map(lambda tp: tp[0] * tp[1],  
                       zip(input_vec, self.weights)))
                , 0.0) + self.bias)

    def train(self, input_vecs, labels, iteration, rate):
        '''
        训练函数：输入训练向量，标签，迭代次数，学习速率
        '''
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)
    
    def _one_iteration(self, input_vecs, labels, rate):
        # 迭代函数
        # 将训练数据打包成元组
        simples = zip(input_vecs, labels)
        for (input_vec, label) in simples:
            output = self.predict(input_vec)
            self._update_weights(input_vec, output, label, rate)
    
    def _update_weights(self, input_vec, output, label, rate):
        delta = label - output
        self.weights = list(map( lambda tp: tp[1] + rate * delta * tp[0], zip(input_vec, self.weights)) ) # HateMath修改
        # 更新bias
        self.bias += rate * delta
    
def f(x):
        '''
        激活函数
        '''
        return 1 if x>0 else 0
    
def get_training_dateset():
        # 基于and真值表构建训练数据
        input_vecs = [[1,1],[0,0],[1,0],[0,1]]
        labels = [1,0,0,0]
        return input_vecs, labels
    
def train_and_perceptron():
        #训练感知器
        p = Perceptron(2, f)
        input_vecs, labels = get_training_dateset()
        p.train(input_vecs, labels, 10, 0.1)
        return p

if __name__ == '__main__': 
    # 训练and感知器
    and_perception = train_and_perceptron()
    # 打印训练获得的权重
    print (and_perception)
    # 测试
    print ('1 and 1 = %d' % and_perception.predict([1, 1]))
    print ('0 and 0 = %d' % and_perception.predict([0, 0]))
    print ('1 and 0 = %d' % and_perception.predict([1, 0]))
    print ('0 and 1 = %d' % and_perception.predict([0, 1]))