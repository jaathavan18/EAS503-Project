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


# Exercise 1
def sphere_area(radius):
    # Write a function that returns the surface area of a sphere given a radius. 
    # Use the input `radius` as the radius
    # round the surface area to two decimales 
    # Use the PI from the math module
    
    import math
    PI = math.pi
    
    # YOUR CODE HERE
    area = 0 # Init
    
    area = 4*radius*radius*PI
    return (round(area,2))


# In[3]:


# Test cell #1 for exercise 1
assert sphere_area(15) == 2827.43
assert sphere_area(7.25) == 660.52


# In[4]:


# Test cell for exercise 1 (hidden)


# In[5]:


# Exercise 2
def sphere_volume(radius):
    # Write a function that returns the volume of a sphere given a radius. 
    # Use the input `radius` as the radius
    # round the surface area to two decimales 
    # Use the PI from the math module
    
    import math
    PI = math.pi
    
    # YOUR CODE HERE
    vol = 0 # Init
    vol = 4/3*PI*(radius**3)
    return (round(vol, 2))


# In[6]:


# Test cell #1 for exercise 2
assert sphere_volume(15) == 14137.17
assert sphere_volume(7.5) == 1767.15


# In[7]:


# Test cell for exercise 2 (Hidden)


# In[8]:


# Exercise 3
def total_wages_for_the_week(hours, wage):
    # Many companies pay time-and-a-half for any hours worked above 40 in a given week.
    # Write a function that uses the inputs to this function, hours worked (hours) and the hourly rate (wage),  and 
    # calculate the total wages for the week. 
    # YOUR CODE HERE
    
    # Init
    total = 0 
    over_time_pay = 0
    
    # Calculate total wages
    total = hours * wage
    ## Calculate extra wages when hours > 40
    if hours > 40 :
        over_time_pay = (hours-40) * wage * 0.5
        total += over_time_pay
        
    return(total)
     
    


# In[9]:


# Test cell for exercise 3
assert total_wages_for_the_week(40, 25) == 1000
assert total_wages_for_the_week(65, 63.4) == 4913.5


# In[10]:


# Test cell for exercise 3 (Hidden)


# In[11]:


# Exercise 4
def convert_quiz_score_to_letter_grade(score):
    # A certain CS professor gives five-point quizzes that are graded on the scale 5-A, 4-B, 3-C, 2-D, 1-F, 0-F.
    # Write a function that accepts a quiz score as an input and uses a decision structure to calculate the corresponding
    # grade. 
    # YOUR CODE HERE
    
    # Calculate grade
    if score == 5:
        return 'A'
    elif score == 4:
        return 'B'
    elif score == 3:
        return 'C'
    elif score == 2:
        return 'D'
    elif score == 1 or score == 0:
        return 'F'
    else:
        pass


# In[12]:


# Test cell  for exercise 4 
assert convert_quiz_score_to_letter_grade(0) == "F"
assert convert_quiz_score_to_letter_grade(1) == "F"
assert convert_quiz_score_to_letter_grade(2) == "D"
assert convert_quiz_score_to_letter_grade(3) == "C"
assert convert_quiz_score_to_letter_grade(4) == "B"
assert convert_quiz_score_to_letter_grade(5) == "A"


# In[13]:


# Exercise 5 
def convert_exam_score_to_letter_grade(score):
    # A certain CS professor gives 100-point exams that are graded on the scale 90-100: A, 80-89: B, 70-79: C,
    # 69-69: D, < 60: F. Write a function that accepts an exam score as input and uses a decision structure 
    # to calculate the corresponding grade. 
    # YOUR CODE HERE
    
    # Calculate grade
    if score < 101 and score > 89 :
        return 'A'
    elif score < 90 and score > 79:
        return 'B'
    elif score < 80 and score > 69:
        return 'C'
    elif score < 70 and score > 59:
        return 'D'
    else:
        return 'F'


# In[14]:


# Test cell  for exercise 5
assert convert_exam_score_to_letter_grade(55) == "F"
assert convert_exam_score_to_letter_grade(35) == "F"
assert convert_exam_score_to_letter_grade(65) == "D"
assert convert_exam_score_to_letter_grade(73) == "C"
assert convert_exam_score_to_letter_grade(87) == "B"
assert convert_exam_score_to_letter_grade(92) == "A"


# In[15]:


# Test cell  for exercise 5 (Hidden)


# In[16]:


# Exercise 6
def calculate_class_standing_from_credits(credits):
    # A certain college classifies students according to credits earned. A student with less than 7 credits is a Freshman.
    # At least 7 credits are required to be a Sophomore, 16 to be a Junior and 26 to be classifed as Senior. 
    # Write a function that calculates class standing from teh number of credits earned. 
    # YOUR CODE HERE
    
    #Calculate class standing from the num of creadits
    if credits > 25 :
        return 'Senior'
    elif credits > 15:
        return 'Junior'
    elif credits > 6:
        return 'Sophomore'
    else:
        return 'Freshman'


# In[17]:


# Test cell  for exercise 6
assert calculate_class_standing_from_credits(6) == "Freshman"
assert calculate_class_standing_from_credits(13) == "Sophomore"
assert calculate_class_standing_from_credits(23) == "Junior"
assert calculate_class_standing_from_credits(45) == "Senior"


# In[18]:


# Test cell  for exercise 6


# In[19]:


# Exercise 7
def calculate_bmi(weight, height):
    # The body mass index (BMI) is calculated as a peron's weight (in pounds) times 720, divided by the square
    # of the person's height (in inches). A BMI in teh rnage 19-25, inclusive, is considered health. Write a function
    # that calculates a person's BMI and prints a message telling whether they are above, within, or below the 
    # the healthy range. 
    # Round to two decimals places
    
    # YOUR CODE HERE
    # Init
    BMI = 0; 
    
    # Calculate BMI
    BMI = (weight*720)/ (height**2)
    BMI = round(BMI,2)
    
    # Consider if the BMI is health
    if BMI > 25:
        print("BMI is above the healthy range")
    elif BMI < 19:
        print("BMI is below the healthy range")
    else:
        print("BMI is within the healthy range")
        
    # return val
    return BMI
    
    


# In[20]:


# Test cell  for exercise 7
assert calculate_bmi(150, 65) == 25.56


# In[21]:


# Test cell  for exercise 7 (hidden)


# In[22]:


# Exercise 8
def calculate_fine(limit, speed):
    # The speeding ticket fine policy in Podunksville is $50 plus $5 for each mph over the limit plus a 
    # penalty of $200 for any speed over 90 mph. Write a function that accepts a speed limit and a clocked
    # speed and either return the string `Legal` or the amount of fine, if the speed is illegal. 

    # YOUR CODE HERE
    # Init
    speeding_ticket = 0
    
    # Check if speed is illegal and compute the speeding ticket fine
    if speed > limit:
        speeding_ticket = 50 + (speed - limit)*5
        if speed > 90:
            speeding_ticket += 200
            
        return speeding_ticket
    else:
        return 'Legal'
    


# In[23]:


# Test cell  for exercise 8
assert calculate_fine(55, 85) == 200
assert calculate_fine(55, 45) == 'Legal'
assert calculate_fine(55, 100) == 475


# In[24]:


# Test cell  for exercise 8


# In[25]:


# Exercise 9
def is_leap_year(year):
    # A year is a leap year if it is divisible by 4, unless it is a century year that is not divisible 
    # by 400. (1800 and 1900 are not leap years while 1600 and 2000 are). Write a function that calculates
    # whether a year is a leap year. Return True or False

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


# In[26]:


# Test cell  for exercise 9
assert is_leap_year(2000) == True
assert is_leap_year(1800) == False
assert is_leap_year(1992) == True


# In[27]:


# Test cell  for exercise 9


# In[28]:


# Exercise 10
def is_palindrome(input_string):
    # Write a function that determines if the input_string is a palindrome. Return True or False
    # Use square brackets to reverse the input_string! Make sure to lower the input string before testing!
    
    # YOUR CODE HERE
    # Init
    lower_str = ''
    reverse_str = ''
    
    # Store the val of lower input string and reverse string
    lower_str = input_string.lower()
    reverse_str = lower_str[::-1]
    
    # Check if the string is a palindrome
    for num_char in range(len(lower_str)):
        if lower_str[num_char] != reverse_str[num_char]:
            return False
        
    return True
        


# In[29]:


# Test cell  for exercise 10
assert is_palindrome("Kayak") == True
assert is_palindrome("Rotator") == True
assert is_palindrome("AACA") == False


# In[30]:


# Test cell  for exercise 10 (hidden)


# In[ ]:




