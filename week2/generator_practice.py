import sys
import os

list = [1,2,3,4,5,6,7,8,9,10]

# for element in list:
#     print(element)

# print(sys.getsizeof(list))  # 136 bytes
# The above is not working when large amount of data is involved
# For example, if we have a list of 1 million elements, the above code will take 1 million iterations to print all the elements.

# The below is working when large amount of data is involved
# for i in range(1,11): # 11 could be 1 million or more
#     print(i)

# print(sys.getsizeof(range(1,11))) # 48 bytes


# mapped_list = map(lambda x: x**2, list)
# map returns a generator object
# print(mapped_list) # [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
# print(sys.getsizeof(mapped_list)) # 48 bytes

# mimic what for loop does
# while True:
#     try:
#         print(next(mapped_list))
#     except StopIteration:
#         print("No more elements")
#         break

# Generator is an elegant way to create an iterator

def gen(list):
    for i in range(list):
        yield i #yield pauses the function and returns the value

n = gen(5)
print(next(n)) # 0
print(next(n)) # 1
print(next(n)) # 2
print(next(n)) # 3
print(next(n)) # 4

# Use case of the generator is when we need to iterate over a large amount of data
# and we don't want to store all the data in memory
# For example, if we need to iterate over a large file,  we don't want to store all the data in memory
# we can use the generator to iterate over the file line by line.
# Filter the words in the file that are longer than 5 characters.

def csv_reader(file):
    with open(file, 'r') as f:
        for line in f:
            yield line

for i, line in enumerate(csv_reader(os.path.join(os.path.dirname(__file__), 'data_example_for_generator.csv'))):
    if i != 0:
        print(line)
