#!/usr/bin/env python
# coding: utf-8

# Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).
# 
# Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

# In[1]:


NAME = "chiehjul"
COLLABORATORS = ""


# ---

# In[2]:


# Ex1
def average_word_len_in_sentence(sentence):
    # Complete this function to calculate the average
    # word length in a sentence
    # Input: sentence
    # Output: average word length in sentence
    # Hint: count punctuations with whatever word they are `touching`
    # Hint: round the average to two decimal places
   
    # YOUR CODE HERE
    # split each word in sentence
    words = sentence.split()

    # Calculate sum of the word length and avg
    sum_words = sum(len(word) for word in words )
    avg = sum_words/len(words)
    
    return round(avg,2)


# In[3]:


assert average_word_len_in_sentence('This is a test sentence!') == 4


# In[4]:


# Hidden test


# In[5]:


#Ex2
def wc(filename):
    # Complete this function to count the number of lines, words, and chars in a file. 
    # Input: filename
    # Output: a tuple with line count, word count, and char count -- in this order

    # YOUR CODE HERE
    # Init
    line_count = 0
    word_count = 0
    char_count = 0
    dif = 0
    
    with open(filename) as file:
        for line in file:
            line_count += 1
            
            words = line.split()
            word_count += len(words)
            char_count += len(line)
                
    return(line_count, word_count, char_count)


# In[6]:


assert wc('ex2_data.txt') == (50, 643, 4298)


# In[7]:


# Ex3a
def is_leap(year):
    # Complete this function to check if year is a leap year
    # Input: year
    # Output: True or False (Boolean)

    # YOUR CODE HERE
    # Check if the year is a leap year
    if year%4 == 0 and year%100 == 0:
        ## If year is divisible by 4 and is a century year
        if year % 400 == 0:
            return True
        return False
    elif year%4 == 0:
        ## If year is divisible by 4 and is not a century year
        return True
    else:
        return False


# In[8]:


assert is_leap(1800) == False
assert is_leap(1900) == False
assert is_leap(1600) == True
assert is_leap(2000) == True


# In[9]:


#Ex3b
def is_date_valid(month, day, year):
    # Complete this function to check if a data is valid, given month, day, and year.  
    # For example, 5/24/1962 is valid, but 9/31/2000 is not
    # Inputs: month, day, year
    # Output: True or False (Boolean)
    # Hint: Use is leap year function

    # YOUR CODE HERE
    
    # Define dict
    months = {
        1: 31,
        2: 28,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 31,
    }    

    if int(month) < 13 and int(month) > 0:
        if int(day) <= months[month] and int(day) > 0:
            return True
        
        # Check Leap year in Feb
        elif int(month) == 2 and int(day) > 0 and is_leap(year) and int(day) <= 29:
            return True
    return False

            


# In[10]:


assert is_date_valid(5, 24, 1962) == True
assert is_date_valid(9, 31, 2000) == False


# In[11]:


# Hidden tests


# In[12]:


#Ex4
def day_number(month, day, year):
    # Complete this function to calculate the day_number given month, day, and year.
    # Information: The days of the year are often numbered from 1 through 365 (or 366).
    # This number can be computed in three steps using int arithmetic:
    # (a) - day_num = 31 * (month - 1) + day
    # (b) - if the month is after February subtract (4*(month)+23)//10
    # (c) - if it's a leap year and after February 29, add 1
    # Hint: First verify the input date is valid, return False if it is not valid; use is_date_valid
    # Hint: Also use the is_leap function
    # Inputs: month, day, year
    # Output: the day number or False (boolean) if the date is invalid. 

    # YOUR CODE HERE
    
    # Define dict and init
    months = {
        1: 31,
        2: 28,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 31,
    }  
    count_day = 0
    
    # Check if the date is valid
    if is_date_valid(month, day, year):
        # Count day without considering leap year
        for mon in range(1, int(month)):
            count_day += months[mon]
        count_day += int(day)
        
        # If is leap && month > 2, day + 1
        if is_leap(year) and int(month) > 2:
            count_day += 1
            
    # Invalid date, return false
    else:
        return False
    
    return count_day


# In[13]:


assert day_number(9, 31, 2000) == False
assert day_number(2, 13, 2020) == 44


# In[14]:


# Hidden tests


# In[15]:


#Ex5
def years_to_double_investment(apr):
    # Complete this function to use a while loop to determine how long it takes for an investment 
    # to double at a given interest rate. The input to this function, apr, is the annualized interest rate
    # and the output is the number of years it takes an investment to double. Note: The amount of the initial 
    # investment (principal) does not matter; you can use $1. 
    # Hint: principal is the amount of money being invested. 
    # apr is the annual percentage rate expressed as a decimal number.  
    # Relationship: value after one year is given by principal * (1+ apr)

    # YOUR CODE HERE
    
    # Init
    year = 0
    principal = 1
    val = principal
    
    # Keep counting value until principal to double
    while val < principal*2:
        val *= (1+apr)
        year += 1
        
    return year


# In[16]:


assert years_to_double_investment(0.06) == 12
assert years_to_double_investment(0.1) == 8


# In[17]:


#Ex6 
def stopping_time(n):
    # Complete this function to return the number of steps taken to reach 1 in
    # the Collatz sequence (https://en.wikipedia.org/wiki/Collatz_conjecture)

    # YOUR CODE HERE
    
    # Init
    step = 0
    cal_n = n
    
    # Stop when val == 1
    while cal_n != 1:
        # Count step
        step += 1
        
        # Collatz sequence
        if cal_n%2 == 0:
            cal_n /= 2
        else:
            cal_n *= 3
            cal_n += 1
    
    return step
    


# In[18]:


assert stopping_time(12) == 9
assert stopping_time(27) == 111


# In[19]:


# hidden tests


# In[20]:


#Ex7
def is_prime(n):
    # A positive whole number > 2 is prime if no number between 2 and sqrt(n)
    # (include) evenly divides n. Write a program that accepts a value of n as
    # input and determine if the value is prime. If n is not prime, your program should
    # return False (boolean) as soon as it finds a value that evenly divides n.
    # Hint: if number is 2, return False

    import math
    
    # YOUR CODE HERE
    
    # Get sqrt num
    divide_range = round(math.sqrt(n))
    
    # If number == 2, return False
    if n == 2:
        return False
    
    # Check if the n == prime
    for divide in range(2,divide_range+1):
        if n % divide == 0:
            return False
        
    return True


# In[21]:


assert is_prime(2) == False
assert is_prime(3) == True
assert is_prime(5) == True
assert is_prime(25) == False


# In[22]:


#Ex8
def all_primes(n):
    # Complete this function to return all the primes as a list less than or equal to n
    # Input: n
    # Output: a list of numbers

    # YOUR CODE HERE
    
    # Init
    primes = []
    
    for num in range(2,n+1):
        if is_prime(num):
            primes.append(num)
            
    return primes


# In[23]:


assert all_primes(5) == [3, 5]
assert all_primes(25) == [3, 5, 7, 11, 13, 17, 19, 23]


# In[24]:


#Ex9
def gcd(m,n):
    # Complete this function to determine the greatest common divisor (GCD). 
    # The GCD of two values can be computed using Euclid's algorithm. Starting with the values
    # m and n, we repeatedly apply the formula: n, m = m, n%m until m is 0. At this point, n is the GCD
    # of the original m and n. 
    # Inputs: m and n which are both natural numbers
    # Output: gcd

    # YOUR CODE HERE
    
    # Init 
    gcd = 0
    dividend = 0
    
    # Let smaller input to be the divisor, otherwise dividend.
    if m < n:
        gcd = m
        dividend = n
    else:
        gcd = n
        dividend = m
    
    # Repeatdely using Euclid's algorithm until dividend%gcd is 0
    while dividend%gcd != 0:
        gcd = (dividend%gcd)
        
    return gcd 


# In[25]:


assert gcd(25,75) == 25
assert gcd(3,13) == 1


# In[ ]:





# In[26]:


#Ex10
def determine_min_max_average(filename):
    # Complete this function to read grades from a file and determine the student with the highest average 
    # test grades and the lowest average test grades. 
    # Input: filename
    # Output: a tuple containing four elements: name of student with highest average, their average, 
    # name of the student with the lowest test grade, and their average. Example ('Student1', 99.50, 'Student5', 65.50)
    # Hint: Round to two decimal places

    # YOUR CODE HERE
    
    #Init
    student_list = [] # student name
    student_dict = {} # student name + avg
    highest = ""      # student name
    lowest = ""       # student name
    
    with open(filename) as file:
        for line in file:
            # skip if line is blannk
            if len(line.strip()) == 0:
                continue
            
            # Split each words by ,
            words = line.strip().split(',')
            
            # Store list and dict
            student_list.append(words[0])
            avg = sum(int(index) for index in words[1:])/(len(words) -1)
            student_dict.update({words[0]:avg})
            
            # Compare with the highest and lowest stud
            if highest == "" and lowest == "":
                highest = words[0]
                lowest = words[0]
            if student_dict[highest] < avg:
                highest = words[0]
            if student_dict[lowest] > avg:
                lowest = words[0]
                
                
    return(highest,student_dict[highest],lowest,student_dict[lowest])
            


# In[27]:


assert determine_min_max_average('ex10_data.txt') == ('student733', 86.2, 'student202', 65.4)

