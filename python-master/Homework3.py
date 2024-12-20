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


# Ex1: Season from Month and Day
def determine_season(month, day):

    # The year is divided into four season: spring, summer, fall (or autumn) and winter.
    # While the exact dates that the seasons change vary a little bit from year to
    # year because of the way that the calender is constructed, we will use the following
    # dates for this exercise:

    # Season  -- First Day
    # Spring  -- March 20
    # Summer  -- June 21
    # Fall  -- September 22
    # Winter    -- December 21

    # Complete this function which takes as its inputs a month and day. It should
    # output the season.
    # input 1: month -- str
    # input 2: day -- int

    # output: month -- str (Spring, Summer, Fall, Winter)

    # YOUR CODE HERE
    # Init
    seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    months = ['January', 'Febuary', 'March', 'April', 'May', 'June', 
              'July', 'August', 'September', 'October','November', 'December']
    
    season_c = 3
    month_c = 1
    
    for mon in range(12):
        if (months[mon] =='March' and day >= 20) or mon > 2:
            season_c = 0
        if (months[mon] =='June' and day >= 21) or mon > 5:
            season_c = 1
        if (months[mon] =='September' and day >= 22) or mon > 8:
            season_c = 2
        if (months[mon] =='December' and day >= 21) or mon > 11:
            season_c = 3
        if months[mon] == month:
            break
            
    return(seasons[season_c])


# In[3]:


assert determine_season('March', 21) == 'Spring'
assert determine_season('June', 21) == 'Summer'
assert determine_season('November', 21) == 'Fall'
assert determine_season('January', 21) == 'Winter'


# In[ ]:





# In[4]:


# Ex2:Is a License Plate Valid?

def is_license_plate_valid(plate):

    # In a particular jurisdiction, older license plates consist of three uppercase
    # letters followed by three digits. When all of the license plates following
    # that pattern had been used, the format was changed to four digits followed by
    # three uppercase letters. 

    # Complete this function whose only input is a license plate and its output
    # is: 1) Older/Valid 2) Newer/Valid 3) Invalid
    # input: plate (str)
    # output: 'Older/Valid' or 'Newer/Valid' or 'Invalid'

    # YOUR CODE HERE
    
    # Check if the plate is older
    if len(plate) == 6:
        # Check if the old plate is valid
        if plate[0:2].isupper() and plate[3:5].isdigit():
            return 'Older/Valid'
        
    # Check if the plate is newer
    elif len(plate) == 7:
        # Check if the new plate is valid
        if plate[0:3].isdigit() and plate[4:6].isupper():
            return 'Newer/Valid'
        
    return 'Invalid'
    


# In[5]:


assert is_license_plate_valid('ABC123') == 'Older/Valid'
assert is_license_plate_valid('GHE952') == 'Older/Valid'
assert is_license_plate_valid('1934ABT') == 'Newer/Valid'
assert is_license_plate_valid('bTR342') == 'Invalid'


# In[ ]:





# In[6]:


#Ex3: Check a password

def check_password(password):

    # In this exercise you will complete this function to determine whether or not
    # a password is good. We will define a good password to be a one that is at least
    # 8 characters long and contains at least one uppercase letter, at least one lowercase
    # letter, and at least one number. This function should return True if the password
    # passed to it as its only parameter is good. Otherwise it should return False. 
    #
    # input: password (str)
    # output: True or False (bool)


    # YOUR CODE HERE
    import re
    
    # Check if len of password is > 8
    if len(password) >= 8:
        # Check if the password is valid
        if re.match(r'.*[A-Z]+.*', password) and re.match(r'.*[a-z]+.*', password) and re.match(r'.*[0-9]+.*', password):
            return True
    return False


# In[7]:


assert check_password('test1234') == False
assert check_password('password123') == False
assert check_password('SuperPasswrd90') == True
assert check_password('letmein!') == False


# In[ ]:





# In[8]:


# Ex4: Magic dates

def is_magic_date(date):
    # A magic date is a date where the day multiplied by the month is equal 
    # to the two digit year. For example, June 10, 1960 is a magic date because
    # June is the sixth month, and 6 times 10 is 60, which is equal to the two
    # digit year. Complete this function to determine whether or not a date is
    # a magic date.

    # input: date (str -- month/day/year)
    # output: True or False (bool)


    # YOUR CODE HERE
    # Get mon, day and year from date
    date_split = date.split('/')
    mon = int(date_split[0])
    day = int(date_split[1])
    year = int(date_split[2])
    
    # Check if the date is magic date.
    if mon*day == year%100:
        return True
    return False


# In[9]:


assert is_magic_date('6/10/1960') == True
assert is_magic_date('6/8/1948') == True


# In[10]:


def remove_outliers(data, num_outliers):
    # When analyzing data collected as a part of a science experiment it 
    # may be desriable to remove the most extreme values before performing
    # other calculations. Complete this function which takes a list of
    # values and an non-negative integer, num_outliers, as its parameters.
    # The function should create a new copy of the list with the num_outliers 
    # largest elements and the num_outliers smallest elements removed. 
    # Then it should return teh new copy of the list as the function's only 
    # result. The order of the elements in the returned list does not have to
    # match the order of the lemetns in the original list.
    # input1: data (list)
    # input2: num_outliers (int)

    # output: list



    # YOUR CODE HERE
    # Create a new copy
    data_sort = data
    # Sort the copy list
    data_sort.sort()
    
    # Remove num_outliers largest and smallest ele
    data_sort = data_sort[num_outliers:]
    data_sort = data_sort[:-(num_outliers)]
    
    return data_sort


# In[11]:


import random
random.seed(1234)
data = [random.randint(50, 150) for ele in range(100)]
data[45] = 1
data[46] = 2
data[90] = 250
data[34] = 300

result = [50, 50, 51, 52, 52, 52, 53, 53, 54, 
55, 55, 55, 55, 56, 58, 58, 59, 59, 59, 60, 61, 61, 
61, 62, 64, 64, 64, 68, 68, 68, 69, 69, 69, 70, 71, 
72, 73, 75, 77, 80, 81, 81, 84, 84, 84, 85, 88, 88, 89, 
92, 94, 94, 95, 95, 98, 103, 106, 108, 108, 109, 109, 109, 
110, 111, 111, 112, 113, 114, 115, 115, 117, 117, 119, 121, 
124, 124, 125, 126, 128, 129, 132, 132, 133, 134, 135, 135, 
135, 136, 138, 140, 141, 148, 148, 149, 149, 150]

assert remove_outliers(data, 2) == result


# In[ ]:





# In[12]:


# Ex6: Removing duplicates

def remove_duplicates(words):
    # Complete this function to remove duplicates from the words list
    # input: words (list)
    # output: a list without duplicates

    # YOUR CODE HERE
    word_wi_dup = []
    
    for word in words:
        if word not in word_wi_dup:
            word_wi_dup.append(word)
    
    return word_wi_dup


# In[13]:


assert remove_duplicates([1,2,3,3,3,4,5,6,7,7,8]) == [1, 2, 3, 4, 5, 6, 7, 8]


# In[ ]:





# In[14]:


# Ex7: List of proper divisors

def proper_divisors(n):
    # A proper divisor ofa  positive integer, n, is a positive integer less than n which divides
    # evenly into n. Complete this function to compute all the proper divisors of a positive
    # integer. The integer is passed to this function as the only parameter. The function will
    # return a list of containing all of the proper divisors as its only reult. 

    # input: n (int)
    # output: list

    # YOUR CODE HERE
    # Init
    proper_div_list = []
    
    # Find proper divisor
    for integer in range(1,n):
        if n % integer == 0:
            proper_div_list.append(integer)

    return proper_div_list


# In[15]:


assert proper_divisors(28) == [1, 2, 4, 7, 14]


# In[ ]:





# In[16]:


#Ex8: Perfect Numbers
def is_number_perfect(n):
    # An integer, n, is said to be perfect when the sum of all of the proper divisors 
    # of n is equal to n. For example, 28 is a perfect number because its proper divisors
    # are 1, 2, 4, 7, and 14 = 28 
    # Complete this function to determine if a the number a perfect number or not. 
    # input: n (int)
    # output: True or False (bool)


    # YOUR CODE HERE
    # Get proper divisor list use the former func
    proper_div_list = proper_divisors(n)
    
    # Check if the sum of proper divisor is same as input
    if n == sum(proper_div_list):
        return True
    
    return False


# In[17]:


assert is_number_perfect(28) == True
assert is_number_perfect(76) == False


# In[ ]:





# In[18]:


# Ex9: Line of Best Fit
def best_line(points):
    # Complete this function to determine the best line. 
    # https://www.varsitytutors.com/hotmath/hotmath_help/topics/line-of-best-fit
    # input: points (list of tuples contain x, y values)
    # output: (m, b) # round both values to two decimal places

    # YOUR CODE HERE
    # Init val
    sum_x = 0
    sum_y = 0
    m = 0
    m_div = 0
    b = 0
    size = len(points)
    
    # Calculate sum of x and y points
    for point in points:
        sum_x += point[0]
        sum_y += point[1]

    # Calculate mean of x and y 
    means_point = (sum_x/size , sum_y/size)
    
    # Calculate m and b
    for point in points:
        m += (point[0]-means_point[0])*(point[1]-means_point[1])
        m_div += (point[0]-means_point[0])**2
    m /= m_div
    b = means_point[1] - m*means_point[0]
    
    # Return the tuples with round to 2 decimal place
    return (round(m,2),round(b,2))
    


# In[19]:


points = (
    (8, 3),
    (2, 10),
    (11, 3),
    (6, 6),
    (5, 8),
    (4, 12),
    (12, 1),
    (9, 4),
    (6, 9),
    (1, 14),
)

assert best_line(points) == (-1.11, 14.08)


# In[ ]:





# In[20]:


# Ex10: Reverse Lookup

def reverse_lookup(input_dict, test_value):
    # Complete this function to find all the keys in a dictionary that map to the input value. 
    # input1: input_dict (dict)
    # input2: test_value
    # output: list of keys

    # YOUR CODE HERE
    # Init
    map_list = []
    
    # Find all the keys that map to the input val
    for key in list(input_dict.keys()):
        if input_dict[key] == test_value:
            map_list.append(key)
            
    return map_list
            


# In[21]:


input_dict = {
    'January': 31,
    'February': 28,
    'March': 31,
    'April': 30,
    'May': 31,
    'June': 30,
    'July': 31,
    'August': 31,
    'September': 30,
    'October': 31,
    'November': 30,
    'December': 31,
}



assert reverse_lookup(input_dict, 31) == ['January', 'March', 'May', 'July', 'August', 'October', 'December']


# In[ ]:




