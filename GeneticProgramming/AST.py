import random
import copy
import numpy.random as npr
import FunctionSet as fs


# author: Martin Irani
# Tarea 4: Programacion Genetica


class AST(object):
    def __init__(self, populationSize, functionSet, termSet, maxDepth, fitnessFunction, inputSize, testdata,
                 mutationRate,
                 treeMutationRate, breedRate):

        self.__dict__.update(locals())
        del self.self

        self.population = [self.generateRandomExpression(-1.0, 1.0) for x in range(populationSize)]
        self.bestFitness = 0

    def __init__(self, populationSize, fitnessarity):
        self.__dict__.update(locals())
        del self.self

    def generateRandomTree(self):
        """
        Method for recursive program generation
        """

        if self.maxDepth is 0 and random.random() > 0.5:
            expr = random.choice(self.termSet)
        else:
            function = random.choice(self.functionSet)
            expr = [self.generateRandomTree(self.functionSet, self.termSet, self.maxDepth)
                    for i in range(function.getArgNum())]
        return expr

    def mutateTree(self, rootNode):
        if random.random() < self.treeMutationRate:
            return self.generateRandomTree()
        else:
            result = copy.deepcopy(rootNode)
            if isinstance(result, FunctionNode):
                result.children = [self.mutateTree(c) for c in rootNode.children]
            return result

    def crossover(self, node1, node2, top=True):
        if random.random() < 0.7 and not top:
            return copy.deepcopy(node2)
        else:
            result = copy.deepcopy(node1)
            if isinstance(node1, FunctionNode) and isinstance(node2, FunctionNode):
                result.children = [self.crossover(c, random.choice(node2.children), False) for c in node1.children]
            return result

    def evaluation(self, rootNode):
        scores = [(self.fitnessFunc(rootNode, self.testData), rootNode) for i in self.population]
        scores.sort()
        return scores

    def getBestFitness(self):
        return self.bestFitness

    def tournamentSelection(self, population, k):
        best = None
        for i in range(k):
            index = population[random.random(1, len(population))]
            if (best is None) or self.evaluation(population) > self.getBestFitness():
                best = index
        return best

    """def evolution(self, gens):
        for gen in range(gens):
            scores = self.evaluation(self.population)
            newpop = [scores[0][1], scores[1][1]]
            self.bestFitness = scores[0][0]
            if self.data != None:
                self.data.append(self.bestFitness)
            while len(newpop) < len(self.population):
                if random.random() < 0.5:
                    newpop.append(self.crossover(self.population[int(npr.beta(1,2) * (len(self.population) - 1))], self.population[int(npr.beta(1,2) * (len(self.population) - 1))]))
                else:
                    newpop.append(self.mutateTree(self.crossover(self.population[int(npr.beta(1,2) * (len(self.population) - 1))], self.population[int(npr.beta(1,2) * (len(self.population) - 1))])))
            self.population = newpop"""


class WrapperFunction:
    def __init__(self, name, function, argNum, var=False):
        """
        Creates a FunctionWrapper object.
        :param name: The name of the function
        :param function: The function that the object wraps
        :param argNum: The number of arguments required by function
        :return:
        """
        self.name = name
        self.function = function
        self.argNum = argNum

        if var is False:
            self.var = False
        else:
            self.var = True

    def getArgNum(self):
        """
        :return: the number of arguments for the function
        """

        if self.var:
            return random.randint(1, self.argNum)
        else:
            return self.argNum


class FunctionNode(object):
    def __init__(self, functionWrapper, children):
        """
        Creates a Node function object

        """
        self.functionWrapper = functionWrapper
        self.children = children

    def evaluate(self, parameters):
        """
        Evaluates this nodes children and then this node with parameters given.
        """
        childrenResults = [child.evaluate(parameters) for child in self.children]
        return self.functionWrapper.function(childrenResults)


class TerminalNode(object):
    def __init__(self, value):
        self.value = value

    def evaluate(self):
        return self.value
