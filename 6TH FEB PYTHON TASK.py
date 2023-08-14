#!/usr/bin/env python
# coding: utf-8

# #  <P style="color:brown"> MD. ARSHAD </p>
# ###  <P style="color:skyblue"> ALISHARMEEN02@GMAIL.COM  </p>

# Q1. Create a function which will take a list as an argument and return the product of all the numbers
# after creating a flat list.
# Use the below-given list as an argument for your function.
# list1 = [1,2,3,4, [44,55,66, True], False, (34,56,78,89,34), {1,2,3,3,2,1}, {1:34, "key2": [55, 67, 78, 89], 4: (45,
# 22, 61, 34)}, [56, 'data science'], 'Machine Learning']
# Note: you must extract numeric keys and values of the dictionary also.

# In[1]:


def flatten_and_calculate_product(input_list):
    def flatten(lst):
        flattened = []
        for item in lst:
            if isinstance(item, list):
                flattened.extend(flatten(item))
            elif isinstance(item, dict):
                for value in item.values():
                    if isinstance(value, (int, float)):
                        flattened.append(value)
                    elif isinstance(value, list):
                        flattened.extend(flatten(value))
            elif isinstance(item, (int, float)):
                flattened.append(item)
        return flattened

    flat_list = flatten(input_list)
    product = 1
    for num in flat_list:
        product *= num
    return product

list1 = [1, 2, 3, 4, [44, 55, 66, True], False, (34, 56, 78, 89, 34), {1, 2, 3, 3, 2, 1}, {1: 34, "key2": [55, 67, 78, 89], 4: (45, 22, 61, 34)}, [56, 'data science'], 'Machine Learning']

result = flatten_and_calculate_product(list1)
print("Product of all numeric values:", result)


# Q2. Write a python program for encrypting a message sent to you by your friend. The logic of encryption
# should be such that, for a the output should be z. For b, the output should be y. For c, the output should
# be x respectively. Also, the whitespace should be replaced with a dollar sign. Keep the punctuation
# marks unchanged.
# Input Sentence: I want to become a Data Scientist.
# Encrypt the above input sentence using the program you just created.

# In[2]:


def encrypt_message(message):
    encrypted = ''
    for char in message:
        if char.isalpha():
            offset = ord('a') if char.islower() else ord('A')
            encrypted_char = chr(25 - (ord(char) - offset) + offset)
            encrypted += encrypted_char
        elif char == ' ':
            encrypted += '$'
        else:
            encrypted += char
    
    return encrypted

input_sentence = "I want to become a Data Scientist."
encrypted_sentence = encrypt_message(input_sentence)
print("Encrypted Message:", encrypted_sentence)


# #  <P style="color:GREEN"> Thank You ,That's All </p>
