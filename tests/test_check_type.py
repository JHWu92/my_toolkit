# coding=utf-8
import unittest
from wjh_other import check_type


class TypeTests(unittest.TestCase):
    def testCheckTypes(self):
        list_int = [1, 2, 3, 4, 5]
        list_with_str = ['1', 2, 3]
        all_int = check_type.check_type(list_int, int)
        with_none_int = check_type.check_type(list_with_str, int)
        int_or_str = check_type.check_type(list_with_str, (int, str))
        self.failUnless(all_int)
        self.failIf(with_none_int)
        self.failUnless(int_or_str)

    def testAllInt(self):
        intable = [1, 2, '3', 4, '5,00', -1, '-10,000']
        digitable_not_intable = intable+['1.', 1.]
        self.failUnless(check_type.all_int_able(intable))
        self.failIf(check_type.all_int_able(digitable_not_intable))

    def testIsInt(self):
        # test cases for numeric type
        self.failUnless(check_type.int_able(1))
        self.failIf(check_type.int_able(1.1))
        self.failIf(check_type.int_able(1.))
        self.failUnless(check_type.int_able(-1))
        self.failIf(check_type.int_able(-1.1))

        # test cases for string
        self.failUnless(check_type.int_able('1'))
        self.failIf(check_type.int_able('1.1'))
        self.failIf(check_type.int_able('1.'))
        self.failUnless(check_type.int_able('1,000'))
        self.failUnless(check_type.int_able('1,00,0'))
        self.failIf(check_type.int_able('1,00,0.0'))

    def testIsDigit(self):
        # test cases for numeric type
        self.failUnless(check_type.digit_able(1))
        self.failUnless(check_type.digit_able(1.1))
        self.failUnless(check_type.digit_able(-1))
        self.failUnless(check_type.digit_able(-1.1))

        # test cases for string
        self.failUnless(check_type.digit_able('1'))
        self.failUnless(check_type.digit_able('-1'))
        self.failUnless(check_type.digit_able('1.1'))
        self.failUnless(check_type.digit_able('1,000'))
        self.failUnless(check_type.digit_able('1,00,0'))
        self.failUnless(check_type.digit_able('1,00,0.0'))
        self.failUnless(check_type.digit_able('-1,00,0.0'))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
