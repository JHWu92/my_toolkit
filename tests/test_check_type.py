# coding=utf-8
import unittest
from wKit.utility import check_dtype as chk


class TypeTests(unittest.TestCase):
    def testCheckTypes(self):
        list_int = [1, 2, 3, 4, 5]
        list_with_str = ['1', 2, 3]
        all_int = chk.chk(list_int, int)
        with_none_int = chk.chk(list_with_str, int)
        int_or_str = chk.chk(list_with_str, (int, str))
        self.failUnless(all_int)
        self.failIf(with_none_int)
        self.failUnless(int_or_str)

    def testAllInt(self):
        intable = [1, 2, '3', 4, '5,00', -1, '-10,000']
        digitable_not_intable = intable+['1.', 1.]
        self.failUnless(chk.all_int_able(intable))
        self.failIf(chk.all_int_able(digitable_not_intable))

    def testIsInt(self):
        # test cases for numeric type
        self.failUnless(chk.int_able(1))
        self.failIf(chk.int_able(1.1))
        self.failIf(chk.int_able(1.))
        self.failUnless(chk.int_able(-1))
        self.failIf(chk.int_able(-1.1))

        # test cases for string
        self.failUnless(chk.int_able('1'))
        self.failIf(chk.int_able('1.1'))
        self.failIf(chk.int_able('1.'))
        self.failUnless(chk.int_able('1,000'))
        self.failUnless(chk.int_able('1,00,0'))
        self.failIf(chk.int_able('1,00,0.0'))

    def testIsDigit(self):
        # test cases for numeric type
        self.failUnless(chk.digit_able(1))
        self.failUnless(chk.digit_able(1.1))
        self.failUnless(chk.digit_able(-1))
        self.failUnless(chk.digit_able(-1.1))

        # test cases for string
        self.failUnless(chk.digit_able('1'))
        self.failUnless(chk.digit_able('-1'))
        self.failUnless(chk.digit_able('1.1'))
        self.failUnless(chk.digit_able('1,000'))
        self.failUnless(chk.digit_able('1,00,0'))
        self.failUnless(chk.digit_able('1,00,0.0'))
        self.failUnless(chk.digit_able('-1,00,0.0'))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
