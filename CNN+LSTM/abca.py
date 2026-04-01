import pandas as pd
import random
import copy
from itertools import combinations
from tqdm import tqdm
import time
import statistics
import os

class QuestionsData:
    def __init__(self, df):
        self.df = df

class Population:
    def __init__(self, pop):
        self.p = pop

    def get_mean(self):
        '''
        获取种群的平均难度级别。
        '''
        return self.p['Difficulty Level'].mean()

    def print_pop(self):
        '''
        打印种群信息。
        '''
        print(self.p)

class Test:
    def __init__(self, diff: float, df, pop_size: int, chap_count: list):
        self.diff = diff # difficulty level required
        self.chap_count = chap_count # number of questions in each chapter
        self.q_num = sum(chap_count) # total number of questions
        self.pop_size = pop_size # population size (for employed, onlookers and scouts)
        self.bank = QuestionsData(df[df['Difficulty Level'] != diff]) # questions with difficulty level different from the required one
        self.population = self.generate_initial_population() # initial population before the algorithm starts
        self.trial = [0] * pop_size # trial limit for each bee
    
    '''
    通过从伙伴中随机选择一个问题替换原始解决方案中的问题（不改变章节ID）来获得新的解决方案。如果新解决方案比原始解决方案更好，则返回 True。否则返回 False。
    '''
    def obtain_new_solution(self, original, partner):
        r1 = original.p[original.p['Group'].isnull()].sample(n = 1)['ID'].values[0]
        
        chapter = original.p[original.p['ID'] == r1]['Chapter_ID'].values[0] # get the chapter ID of the random question
        
        new_question = partner.p[partner.p['Chapter_ID'] == chapter].sample(n = 1) # get a random question from the partner (with the same chapter)

        gr = new_question['Group'].values[0]
        if gr != gr: # check if the question has group
            return False, None
                
        temp = copy.deepcopy(original)
        temp.p[temp.p['ID'] == r1] = new_question.iloc[0]
        
        if temp.p.nunique()["ID"] != self.q_num: 
            return False, None

        if abs(temp.get_mean() - self.diff) < abs(original.get_mean() - self.diff): # if the new solution is better than the original one
            return True, temp
        else:
            return False, None

    '''
    根据章节计数要求，随机生成一个解决方案。
    '''
    def generate_one_solution(self):
        flag = False
        gr = [1,2,3,4] # group IDs
        df = pd.DataFrame()
        ch = random.choice(list(combinations(gr, random.randint(1, len(gr))))) # a random combination of group (example: (1,2), (2,4), ...)
        if random.random() < 0.4: # choosing a test with group
            df = self.bank.df[self.bank.df['Group'].isin(ch)]
            for i,j in zip(self.chap_count, range(len(self.chap_count))): # if number of chapter is larger than required, skip and generate a solution without group
                if i < df[df['Chapter_ID'] == j + 1].shape[0]:
                    flag = True
                    break

            if not flag:
                for i,j in zip(self.chap_count, range(len(self.chap_count))):
                    count = df[df['Chapter_ID'] == j + 1].shape[0]
                    if count == i: continue
                    df = pd.concat([df, self.bank.df[(self.bank.df['Group'].isnull()) & (self.bank.df['Chapter_ID'] == j + 1)].sample(n = i-count)], ignore_index=True, axis=0).sort_values(by=['Chapter_ID'])

                return df

        df = pd.DataFrame()
        for i,j in zip(self.chap_count, range(len(self.chap_count))):
            df = pd.concat([df, self.bank.df[(self.bank.df['Group'].isnull()) & (self.bank.df['Chapter_ID'] == j + 1)].sample(n = i)], axis=0, ignore_index=True)
        
        return df

    '''
    创建初始种群。
    '''
    def generate_initial_population(self):
        return [Population(self.generate_one_solution()) for _ in range(self.pop_size)]

    '''
    获取整个种群中各个解决方案的平均难度。
    '''
    def get_all_mean(self):
        return [i.get_mean() for i in self.population]

    '''
    获取整个种群的适应度值列表（平均难度与目标难度之间的绝对差值）。
    '''
    def get_all_fitness_values(self):
        return list(map(lambda x: abs(x - self.diff), self.get_all_mean()))

    '''
    启动雇佣蜂阶段。种群中的每个解决方案与另一个随机选择的解决方案配对以生成新解决方案。
    如果新解决方案比原方案好，则替换它并将试验次数设为0。否则，该方案的试验次数加1。
    '''
    def deploy_employed(self):
        self.trial = [0] * self.pop_size
        for i,j in zip(self.population, range(len(self.population))):
            idx = list(range(len(self.population)))
            idx.pop(j)
            partner = self.population[random.choice(idx)]
            ret, tm = self.obtain_new_solution(i, partner)
            if ret: 
                self.trial[j] = 0
                self.population[j] = tm
            else: 
                self.trial[j] += 1
                del tm

    '''
    启动观察蜂阶段。与雇佣蜂阶段类似，但每个解决方案根据其适应度拥有一定的概率被选中来生成新解决方案。
    该阶段一直进行，直到生成的新解决方案数量等于种群大小。
    '''
    def deploy_onlookers(self):
        i = 0
        t = 0
        total = sum(self.get_all_fitness_values())
        prob = list(map(lambda x: x / total, self.get_all_fitness_values())) # calculate all probabilities
        while (t < self.pop_size):
            p = random.random()
            if p < prob[i]:
                idx = list(range(len(self.population)))
                idx.pop(i)
                partner = self.population[random.choice(idx)]
                ret, tm = self.obtain_new_solution(self.population[i], partner)
                if ret: 
                    self.trial[i] = 0
                    self.population[i] = tm
                else:
                    self.trial[i] += 1
                    del tm
                t += 1
            i += 1
            i = i % self.pop_size
            
    '''
    启动侦察蜂阶段。如果某个解决方案的试验次数达到上限，则该解决方案会被随机生成的新解决方案替换。
    '''
    def deploy_scouts(self, limit):
        for i in range(len(self.trial)):
            if i > float(limit):
                self.trial[i] = 0
                self.population[i] = Population(self.generate_one_solution())

    '''
    用于打印种群数据框的辅助函数。
    '''
    def print_population(self):
        for i in self.population:
            i.print_pop()

    '''
    用于打印当前最佳解决方案及其平均难度值的辅助函数。
    '''
    def print_best_solution(self):
        best = sorted(self.population, key = lambda x: abs(self.diff - x.get_mean()))[0]
        best.print_pop()
        print('Best solution: Test', self.population.index(best) + 1)
        print("Difficulty:", best.get_mean())
        return self.population.index(best)

# def change_state(is_stopped, count_stop, i):
#     for j in range(len(count_stop)):
#         if is_stopped[j] == False:
#             if count_stop[j] > args['stop_iters']:
#                 is_stopped[j] = i
#     return is_stopped
        
# def check_stop_condition(output, count_stop, is_stopped):
#     a = output.iloc[-1].tolist()
#     b = output.iloc[-2].tolist()
#     a,b = np.array(a), np.array(b)
#     result = abs((a-b)/b) < args['stop_threshold']
#     for i in range(len(count_stop)):
#         if is_stopped[i] == False:
#             if result[i]:
#                 count_stop[i] += 1
#             else:
#                 count_stop[i] = 0
#     return count_stop
    
def print_output(test, start, output, i = None):
    '''
    打印输出结果，包括标准差、总运行时间以及平均每次迭代运行时间，并将结果保存到 CSV 文件中。
    '''
    best_idx = test.print_best_solution() # print the best solution with its average difficulty

    print("Standard deviation:", statistics.pstdev(test.get_all_fitness_values()))
    print('Total runtime:', time.time() - start, 'seconds')
    print('Average runtime:', (time.time() - start) / i, 'seconds per iteration')

    if args['save']:
        output.columns=['Test {i}'.format(i=i) for i in range(1, args['pop_size'] + 1)]
        output.round(4).to_csv(args['output'])
    return best_idx

def iteration_phase(test, output):
    '''
    执行指定次数的算法迭代阶段，分别调用雇佣蜂、观察蜂和侦察蜂，并将适应度结果存储在输出中。
    '''
    start = time.time()

    for i in tqdm(range(args['num_iters'])): # iterate the algorithm
        test.deploy_employed() # deploy employed bees
        test.deploy_onlookers() # deploy onlookers
        test.deploy_scouts(args['limit']) # deploy scouts
        output = pd.concat([output, pd.DataFrame(test.get_all_fitness_values()).transpose()], axis = 0, ignore_index=True)
#         count_stop = check_stop_condition(output, count_stop, is_stopped)
#         is_stopped = change_state(is_stopped, count_stop, i+1)

    best_idx = print_output(test, start, output, i + 1)
    return output, best_idx



#####
if __name__ == '__main__':

    # Settings
    base_dir = os.path.dirname(os.path.abspath(__file__))
    args = { 
        "file": os.path.join(base_dir, "Main_1000_2.csv"), # Question file directory
        "diff": 0.52, # Difficulty required
        "pop_size": 20, # Population size for each phase
        "num_iters": 100, # Number of iterations
        "threshold_limit_iters": 20, # Limit for threshold
        "threshold": 0.0005, # Threshold value
        "limit": 20, # Limitation for scout phase
        'save': True, # Save fitness values to a csv file
        'output': 'output.csv', # Output file name
    }

    unique_chap = pd.read_csv(args['file']).nunique()['Chapter_ID']
    args['section'] = [100 // unique_chap] * unique_chap

    # Main flow
    df = pd.read_csv(args['file'])
    test = Test(args['diff'], df, args['pop_size'], args['section']) # initialize the test

    # count_stop = [0] * args['pop_size']
    # is_stopped = [False] * args['pop_size']

    output = pd.DataFrame()
    output = pd.concat([output, pd.DataFrame(test.get_all_fitness_values()).transpose()], axis = 0)
    i = 0

    best_idx = None

    init_pop = test.population
    start = time.time()

    while True:
        test.deploy_employed()
        test.deploy_onlookers()
        test.deploy_scouts(args['limit'])
        i += 1
        output = pd.concat([output, pd.DataFrame(test.get_all_fitness_values()).transpose()], axis = 0, ignore_index=True)
    #     count_stop = check_stop_condition(output, count_stop, is_stopped)
    #     is_stopped = change_state(is_stopped, count_stop, i)

        if any(map(lambda x: x < args['threshold'], test.get_all_fitness_values())):
            best_idx = print_output(test, start, output, i)
            break
            
        if i > args['threshold_limit_iters']:
    #         count_stop = [0] * args['pop_size']
    #         is_stopped = [False] * args['pop_size']
            test.population = init_pop
            output = pd.DataFrame()
            output = pd.concat([output, pd.DataFrame(test.get_all_fitness_values()).transpose()], axis = 0)
            output, best_idx = iteration_phase(test, output)
            break
            
    test.population = init_pop
    print('Average initial fitness value:', statistics.mean(test.get_all_fitness_values()))