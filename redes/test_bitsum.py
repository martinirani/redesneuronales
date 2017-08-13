from unittest import TestCase
from bitsum import Bitsum


class TestBitsum(TestCase):

    def setUp(self):

        self.y0 = 0
        self.y1 = 1

    def testAnd(self):
        bs1 = Bitsum(self.y0, self.y0)
        self.assertEqual(bs1.outputsum(), "00")

        bs2 = Bitsum(self.y0, self.y1)
        self.assertEqual(bs2.outputsum(), "01")

        bs3 = Bitsum(self.y1, self.y1)
        self.assertEqual(bs3.outputsum(), "10")




