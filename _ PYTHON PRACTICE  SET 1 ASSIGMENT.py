#!/usr/bin/env python
# coding: utf-8

# #  <P style="color:brown"> MD. ARSHAD </p>
# ###  <P style="color:skyblue"> ALISHARMEEN02@GMAIL.COM  </p>

# Q1. Write a Python program that defines a function called "add_numbers" that takes two arguments (i.e.,
# numbers) and returns their sum. Within the function, add the two numbers together and return the result
# using the return statement. Call the function with the values 5 and 6, and print out the returned result.
# This will result in the addition of 5 and 6, with the output of the program being the sum of these two
# numbers.

# In[2]:


#You can create the Python program as follows to define the "add_numbers" function, call 
#it with the values 5 and 6, and print out the result
# Define the add_numbers function
def add_numbers(num1, num2):
    result = num1 + num2
    return result

# Call the function with values 5 and 6
sum_result = add_numbers(5, 6)

# Print the returned result
print("The sum of 5 and 6 is:", sum_result)


# In[ ]:




Q2. Write a Python program that calculates the square root of a given number using a built-in function.
Specifically, the program should take an integer or float input from the user, calculate its square root
using the 'sqrt()' function from the 'math' module, and print out the result to the user. As an example,
calculate the square root of the number 625 using this program, which should output the value of 25.
# In[3]:


# Import the math module to use the sqrt() function
import math

# Get user input for the number
user_input = float(input("Enter a number to calculate its square root: "))

# Calculate the square root using math.sqrt()
square_root = math.sqrt(user_input)

# Print the result
print(f"The square root of {user_input} is: {square_root}")


# In[ ]:




Q3.Write a program that prints all prime numbers between 0 to 50.
# In[4]:


#W Function to check if a number is prime
def is_prime(num):
    if num <= 1:
        return False
    if num <= 3:
        return True
    if num % 2 == 0 or num % 3 == 0:
        return False
    i = 5
    while i * i <= num:
        if num % i == 0 or num % (i + 2) == 0:
            return False
        i += 6
    return True

# Print prime numbers between 0 and 50
print("Prime numbers between 0 and 50:")
for number in range(51):
    if is_prime(number):
        print(number, end=" ")


# In[ ]:





# Q4.How can we swap the values of three variables (let's say a, b, and c) without using a fourth variable?
# For example, if we have a=5, b=8, and c=9, how can we obtain a=9, b=5, and c=8? The challenge is to
# perform this operation without using an additional variable to store any of the values during the
# swapping process.

# In[5]:


a  = 5
b = 8
c = 9

# Swap the values of a, b, and c
a = a + b + c  # a now holds the sum of all three values (5 + 8 + 9 = 22)
b = a - (b + c)  # b now holds the value of a before the swap (22 - 8 - 9 = 5)
c = a - (b + c)  # c now holds the value of b before the swap (22 - 5 - 9 = 8)
a = a - (b + c)  # a now holds the value of c before the swap (22 - 5 - 8 = 9)

# Now, a, b, and c have been swapped
print("a =", a)
print("b =", b)
print("c =", c)


# In[ ]:





# Q5. Can you write a program that determines the nature of a given number (in this case, 87) as being
# positive, negative, or zero? The program should be designed to take the number as input and perform the
# necessary calculations to determine if the number is positive (i.e., greater than zero), negative (i.e., less
# than zero), or zero (i.e., equal to zero). The output of the program should indicate which of these three
# categories the given number falls into.

# In[6]:


# Get user input for the number
number = float(input("Enter a number: "))

# Check the nature of the number
if number > 0:
    print("The number is positive.")
elif number < 0:
    print("The number is negative.")
else:
    print("The number is zero.")


# In[7]:


# Get user input for the number
number = float(input("Enter a number: "))

# Check the nature of the number
if number > 0:
    print("The number is positive.")
elif number < 0:
    print("The number is negative.")
else:
    print("The number is zero.")


# In[8]:


# Get user input for the number
number = float(input("Enter a number: "))

# Check the nature of the number
if number > 0:
    print("The number is positive.")
elif number < 0:
    print("The number is negative.")
else:
    print("The number is zero.")


# In[ ]:





# Q6. How can you create a program that determines whether a given number (in this case, 98) is even or
# odd? The program should be designed to take the number as input and perform the necessary
# calculations to determine whether it is divisible by two. If the number is divisible by two without leaving a
# remainder, it is an even number, and if there is a remainder, it is an odd number. The output of the
# program should indicate whether the given number is even or odd.

# In[9]:


number = int(input("Enter a number: "))

# Check if the number is even or odd
if number % 2 == 0:
    print(f"{number} is an even number.")
else:
    print(f"{number} is an odd number.")


# In[10]:


# Get user input for the number
number = int(input("Enter a number: "))

# Check if the number is even or odd
if number % 2 == 0:
    print(f"{number} is an even number.")
else:
    print(f"{number} is an odd number.")


# In[ ]:





# Q7.Write a program for sum of digits.the digits are 76543 and the output should be 25.

# In[11]:


number = 76543

# Initialize a variable to store the sum of digits
sum_of_digits = 0

# Iterate through each digit of the number
while number > 0:
    digit = number % 10  # Get the last digit
    sum_of_digits += digit  # Add the digit to the sum
    number //= 10  # Remove the last digit from the number

# Print the sum of digits
print("The sum of digits is:", sum_of_digits)


# In[ ]:





# Q8.Write a program for reversing the given number 5436 and the output should be 6345.

# In[12]:


number = 5436

# Initialize a variable to store the reversed number
reversed_number = 0

# Reverse the number
while number > 0:
    # Get the last digit of the number
    digit = number % 10
    
    # Add the digit to the reversed number (shift the reversed number to the left and add the digit)
    reversed_number = reversed_number * 10 + digit
    
    # Remove the last digit from the original number
    number //= 10

# Print the reversed number
print("The reversed number is:", reversed_number)


# In[ ]:





# Q9.Write a program to check if a given number 371 is an Armstrong number?

# In[13]:


number = 371

# Convert the number to a string to find its number of digits
num_str = str(number)

# Calculate the number of digits
num_digits = len(num_str)

# Initialize a variable to store the sum of digits raised to the power of num_digits
sum_of_powers = 0

# Calculate the sum of digits raised to the power of num_digits
for digit_char in num_str:
    digit = int(digit_char)
    sum_of_powers += digit ** num_digits

# Check if the number is an Armstrong number
if sum_of_powers == number:
    print(f"{number} is an Armstrong number.")
else:
    print(f"{number} is not an Armstrong number.")


# In[ ]:





# Q10.Write a program the given year is 1996, a leap year.

# In[14]:


year = 1996

# Check if it's a leap year
if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
    print(f"{year} is a leap year.")
else:
    print(f"{year} is not a leap year.")


# In[ ]:





# #  <P style="color:GREEN"> Thank You ,That's All </p>
