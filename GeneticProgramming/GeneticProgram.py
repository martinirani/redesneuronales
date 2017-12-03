import AST as ast
import FunctionSet as fs
import matplotlib.pyplot as mpl


# author: Martin Irani
# Tarea 4: Programacion Genetica

def example1:
    populationSize =
    functionSet = [fs.addw, fs.divw, fs.mulw, fs.subw]
    termSet = ['x', 'y', 'z']
    maxDepth = 3
    arity = 2

    asTree = AST(populationSize, functionSet, termSet, maxDepth,
                 fitnessFunction, inputSize, testdata, mutationRate, treeMutationRate, breedRate)
    output = asTree.generateRandomExpression(functionSet, termSet, maxDepth, 'full')


def example2:
