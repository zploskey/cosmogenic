import unittest

class TestBase(unittest.TestCase):

    def monotonically_decreasing(self, p):
        dp = (p[:-1] - p[1:])
        decreasing = (dp >= 0).all()
        self.assertTrue(decreasing, "p: " + str(p) + " dp: " + str(dp))
