# 1. abs()
num = -10
print("Absolute value of -10:", abs(num))

# 2. all()
lst = [True, True, False]
print("All elements are True:", all(lst))

# 3. any()
print("Any element is True:", any(lst))

# 4. ascii()
print("ASCII representation of 'a':", ascii('a'))

# 5. bin()
print("Binary representation of 10:", bin(10))

# 6. bool()
print("Boolean value of 0:", bool(0))

# 7. bytearray()
byte_arr = bytearray([65, 66, 67])
print("Bytearray:", byte_arr)

# 8. bytes()
byte_str = bytes([65, 66, 67])
print("Bytes:", byte_str)

# 9. callable()
def func():
    return True
print("Is 'func' callable:", callable(func))

# 10. chr()
print("Character with ASCII code 65:", chr(65))

# 11. classmethod()
class MyClass:
    @classmethod
    def my_method(cls):
        print("Class method called")
MyClass.my_method()

# 12. compile()
code = "print('Hello, world!')"
compiled_code = compile(code, filename='', mode='exec')
exec(compiled_code)

# 13. complex()
complex_num = complex(2, 3)
print("Complex number:", complex_num)

# 14. delattr()
class MyClass:
    x = 5
obj = MyClass()
print("Before deletion:", hasattr(obj, 'x'))
delattr(MyClass, 'x')
print("After deletion:", hasattr(obj, 'x'))

# 15. dict()
my_dict = dict(name='John', age=30)
print("Dictionary:", my_dict)

# 16. dir()
print("Directory of current module:", dir())

# 17. divmod()
quotient, remainder = divmod(10, 3)
print("Quotient and remainder:", quotient, remainder)

# 18. enumerate()
lst = ['a', 'b', 'c']
for index, value in enumerate(lst):
    print("Index:", index, "Value:", value)

# 19. eval()
result = eval("2 + 3 * 5")
print("Result of evaluation:", result)

# 20. exec()
exec("print('Hello from exec()')")

# 21. filter()
def is_even(x):
    return x % 2 == 0
even_nums = filter(is_even, [1, 2, 3, 4, 5])
print("Even numbers:", list(even_nums))

# 22. float()
print("Float value of '3.14':", float('3.14'))

# 23. format()
print("Formatted value:", format(123.456, '.2f'))

# 24. frozenset()
frozen_set = frozenset([1, 2, 3])
print("Frozen set:", frozen_set)

# 25. getattr()
class MyClass:
    attr = "Hello"
obj = MyClass()
print("Value of 'attr':", getattr(obj, 'attr'))

# 26. globals()
print("Global variables:", globals())

# 27. hasattr()
class MyClass:
    x = 5
obj = MyClass()
print("Does 'obj' have attribute 'x'?", hasattr(obj, 'x'))

# 28. hash()
print("Hash of 'hello':", hash('hello'))

# 29. help()
print("Help for 'print' function:")
help(print)

# 30. hex()
print("Hexadecimal representation of 255:", hex(255))

# 31. id()
x = 10
print("ID of x:", id(x))

# 32. input()
name = input("Enter your name: ")
print("Hello,", name)

# 33. int()
print("Integer value of '10':", int('10'))

# 34. isinstance()
print("Is '10' an instance of int?", isinstance(10, int))

# 35. issubclass()
class MyClass:
    pass
class MySubClass(MyClass):
    pass
print("Is 'MySubClass' a subclass of 'MyClass'?", issubclass(MySubClass, MyClass))

# 36. iter()
my_list = [1, 2, 3]
my_iter = iter(my_list)
print("Iterator:", next(my_iter))

# 37. len()
print("Length of [1, 2, 3]:", len([1, 2, 3]))

# 38. list()
print("List from range(5):", list(range(5)))

# 39. locals()
print("Local variables:", locals())

# 40. map()
def square(x):
    return x ** 2
squared_nums = map(square, [1, 2, 3, 4, 5])
print("Squared numbers:", list(squared_nums))

# 41. max()
print("Maximum of [1, 2, 3, 4, 5]:", max([1, 2, 3, 4, 5]))

# 42. memoryview()
my_bytes = b'Hello'
my_view = memoryview(my_bytes)
print("Memory view:", my_view)

# 43. min()
print("Minimum of [1, 2, 3, 4, 5]:", min([1, 2, 3, 4, 5]))

# 44. next()
my_iter = iter([1, 2, 3])
print("Next element of iterator:", next(my_iter))

# 45. object()
class MyClass:
    pass
obj = MyClass()
print("Object:", obj)

# 46. oct()
print("Octal representation of 8:", oct(8))

# 47. open()
file = open('example.txt', 'w')
file.write("Hello, world!")
file.close()

# 48. ord()
print("ASCII value of 'A':", ord('A'))

# 49. pow()
print("2 to the power of 3:", pow(2, 3))

# 50. print()
print("Hello, world!")

# 51. property()
class MyClass:
    def __init__(self):
        self._x = None
    def get_x(self):
        return self._x
    def set_x(self, value):
        self._x = value
    x = property(get_x, set_x)
obj = MyClass()
obj.x = 5
print("Value of 'x':", obj.x)

# 52. range()
print("Range from 0 to 5:", list(range(5)))

# 53. repr()
print("Representation of 'hello':", repr('hello'))

# 54. reversed()
reversed_lst = reversed([1, 2, 3])
print("Reversed list:", list(reversed_lst))

# 55. round()
print("Rounded value of 3.14159:", round(3.14159, 2))

# 56. set()
print("Set from [1, 2, 3]:", set([1, 2, 3]))

# 57. setattr()
class MyClass:
    pass
obj = MyClass()
setattr(obj, 'x', 5)
print("Value of 'x':", obj.x)

# 58. slice()
my_list = [1, 2, 3, 4, 5]
my_slice = slice(2)
print("Slice of [1, 2, 3, 4, 5]:", my_list[my_slice])

# 59. sorted()
print("Sorted list:", sorted([5, 2, 8, 1, 3]))

# 60. staticmethod()
class MyClass:
    @staticmethod
    def my_method():
        print("Static method called")
MyClass.my_method()

# 61. str()
print("String value of 123:", str(123))

# 62. sum()
print("Sum of [1, 2, 3, 4, 5]:", sum([1, 2, 3, 4, 5]))

# 63. super()
class MyParentClass:
    def __init__(self):
        print("Parent class constructor called")
class MyChildClass(MyParentClass):
    def __init__(self):
        super().__init__()
obj = MyChildClass()

# 64. tuple()
print("Tuple from [1, 2, 3]:", tuple([1, 2, 3]))

# 65. type()
print("Type of 'hello':", type('hello'))

# 66. vars()
class MyClass:
    x = 5
print("Class variables:", vars(MyClass))

# 67. zip()
zipped = zip([1, 2, 3], ['a', 'b', 'c'])
print("Zipped list:", list(zipped))

# 68. __import__()
import math
print("Imported math module:", math)

# 69. locals()
print("Local variables:", locals())

# 70. abs()
print("Absolute value of -10:", abs(-10))

# 71. round()
print("Rounded value of 3.14159:", round(3.14159))

# 72. pow()
print("2 raised to the power of 3:", pow(2, 3))

# 73. divmod()
quotient, remainder = divmod(10, 3)
print("Quotient and remainder:", quotient, remainder)

# 74. hex()
print("Hexadecimal representation of 255:", hex(255))

# 75. oct()
print("Octal representation of 8:", oct(8))

# 76. bin()
print("Binary representation of 10:", bin(10))

# 77. ascii()
print("ASCII representation of 'a':", ascii('a'))

# 78. ord()
print("ASCII value of 'A':", ord('A'))

# 79. chr()
print("Character with ASCII code 65:", chr(65))

# 80. sum()
print("Sum of [1, 2, 3, 4, 5]:", sum([1, 2, 3, 4, 5]))

# 81. max()
print("Maximum of [1, 2, 3, 4, 5]:", max([1, 2, 3, 4, 5]))

# 82. min()
print("Minimum of [1, 2, 3, 4, 5]:", min([1, 2, 3, 4, 5]))

# 83. len()
print("Length of [1, 2, 3]:", len([1, 2, 3]))

# 84. format()
print("Formatted value:", format(123.456, '.2f'))

# 85. sorted()
print("Sorted list:", sorted([5, 2, 8, 1, 3]))

# 86. reversed()
reversed_lst = reversed([1, 2, 3])
print("Reversed list:", list(reversed_lst))

# 87. enumerate()
lst = ['a', 'b', 'c']
for index, value in enumerate(lst):
    print("Index:", index, "Value:", value)

# 88. slice()
my_list = [1, 2, 3, 4, 5]
my_slice = slice(2)
print("Slice of [1, 2, 3, 4, 5]:", my_list[my_slice])

# 89. map()
def square(x):
    return x ** 2
squared_nums = map(square, [1, 2, 3, 4, 5])
print("Squared numbers:", list(squared_nums))

# 90. filter()
def is_even(x):
    return x % 2 == 0
even_nums = filter(is_even, [1, 2, 3, 4, 5])
print("Even numbers:", list(even_nums))

# 91. zip()
zipped = zip([1, 2, 3], ['a', 'b', 'c'])
print("Zipped list:", list(zipped))

# 92. all()
lst = [True, True, False]
print("All elements are True:", all(lst))

# 93. any()
print("Any element is True:", any(lst))

# 94. bool()
print("Boolean value of 0:", bool(0))

# 95. int()
print("Integer value of '10':", int('10'))

# 96. float()
print("Float value of '3.14':", float('3.14'))

# 97. str()
print("String value of 123:", str(123))

# 98. list()
print("List from range(5):", list(range(5)))

# 99. tuple()
print("Tuple from [1, 2, 3]:", tuple([1, 2, 3]))

# 100. set()
print("Set from [1, 2, 3]:", set([1, 2, 3]))
