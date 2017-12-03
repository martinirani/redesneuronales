from AST import WrapperFunction


# Set of functions
# Tarea 04 - Programacion genetica
# Autor: Martin Irani

def addition(arg1, arg2):
    return arg1 + arg2


def substraction(arg1, arg2):
    return arg1 - arg2


def multiplication(arg1, arg2):
    return arg1 * arg2


def division(arg1, arg2):
    if arg2 == 0:
        return 1
    else:
        return arg1 / arg2


addw = WrapperFunction("+", addition, 2)
subw = WrapperFunction("-", substraction, 2)
mulw = WrapperFunction("*", multiplication, 2)
divw = WrapperFunction("/", division, 2)
