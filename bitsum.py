from numpy import array, dot
from Perceptron import PerceptronNand

#

class Bitsum:

    def __init__(self, y, z):
        self.y = y
        self.z = z

    def outputsum(self):

        # layer 1
        yz = array([self.y, self.z])
        layer1 = PerceptronNand(yz).output()

        # layer 2

        input_layer2A = array([self.y, layer1])
        layer2A = PerceptronNand(input_layer2A).output()

        input_layer2B = array([self.z, layer1])
        layer2B = PerceptronNand(input_layer2B).output()

        # layer 3

        input_sum = array([layer2A, layer2B])
        input_carry = array([layer1, layer1])

        output_sum = PerceptronNand(input_sum).output()
        output_carry = PerceptronNand(input_carry).output()

        return (str(output_carry) + str(output_sum))


