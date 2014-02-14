import unittest

class TestBase(unittest.TestCase):

    def monotonically_decreasing(self, p):
        decreasing =((p[:-1] - p[1:]) >= 0).all()
        self.assertTrue(decreasing)
