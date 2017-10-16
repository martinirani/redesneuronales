from unittest import TestCase
from Perceptron import PerceptronNand, PerceptronAnd, PerceptronOr

class TestPerceptron(TestCase):

    def setUp(self):

        self.x1 = [0, 0]
        self.x2 = [1, 0]
        self.x3 = [0, 1]
        self.x4 = [1, 1]

    def testAnd(self):
        pAnd1 = PerceptronAnd(self.x1)
        self.assertEqual(pAnd1.output(), 0)

        pAnd2 = PerceptronAnd(self.x2)
        self.assertEqual(pAnd2.output(), 0)

        pAnd3 = PerceptronAnd(self.x3)
        self.assertEqual(pAnd3.output(), 0)

        pAnd4 = PerceptronAnd(self.x4)
        self.assertEqual(pAnd4.output(), 1)

    def testOr(self):
        pOr1 = PerceptronOr(self.x1)
        self.assertEqual(pOr1.output(), 0)

        pOr2 = PerceptronOr(self.x2)
        self.assertEqual(pOr2.output(), 1)

        pOr3 = PerceptronOr(self.x3)
        self.assertEqual(pOr3.output(), 1)

        pOr4 = PerceptronOr(self.x4)
        self.assertEqual(pOr4.output(), 1)

    def testNand(self):
        pNand1 = PerceptronNand(self.x1)
        self.assertEqual(pNand1.output(), 1)

        pNand2 = PerceptronNand(self.x2)
        self.assertEqual(pNand2.output(), 1)

        pNand3 = PerceptronNand(self.x3)
        self.assertEqual(pNand3.output(), 1)

        pNand4 = PerceptronNand(self.x4)
        self.assertEqual(pNand4.output(), 0)
