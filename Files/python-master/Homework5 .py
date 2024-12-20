#!/usr/bin/env python
# coding: utf-8

# Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).
# 
# Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

# In[1]:


NAME = "chiehjul"
COLLABORATORS = ""


# ---

# # Ex 1 - Handling Exceptions (10 Points)
# In this problem you will read from a text file (**data_p2.txt**) included in the assignment. Please download it and store it in the same directory as this notebook. The code for reading data from the file is given below. You need to store the data and perform a simple mathematical operation on each data point. Each line is _supposed_ to contain a floating point number. But what you will observe is that some lines might have erroneous entries. You need to ignore those lines (Hint: Use `Exception` handling). 
# 
# The idea is to implement a function, `computeMedian()`, which reads in a file and computes the median of the numbers and returns the output. You may use the inbuilt function `sort` when computing the median.
# 
# _DO NOT USE ANY INBUILT OR OTHER FUNCTION TO DIRECTLY COMPUTE MEDIAN_

# In[2]:


# Reading from a file. The following code just reads in every line in a text file without doing anything with it.
# You can use the code in your solution. DO NOT CALL this function in your implementation.
def dummyFileReader():
    with open('data_p2.txt','r') as f: #we open the file in 'read' mode. The 'with' clause is similar to 'finally' clause
        for line in f: #iterate over the file line by line
            s = line.strip() #strip() removes the endline character at the end of the line. Line is of type 'str'

def computeMedian():
    """
    The function has no input parameter, return median value of the list
    """
    # YOUR CODE HERE
    import re
    
    # Init
    val_list = []

    # Read file
    with open('data_p2.txt','r') as f:
        for line in f:
            # Remove the endline char
            val = line.strip()
            # Append val in list with float type
            try:
                val_list.append(float(val))
            except:
                continue
    
    # Sort the list
    val_list = sorted(val_list)
    
    # Return the median val
    return val_list[round(len(val_list)/2)]


# In[3]:


assert computeMedian() == 0.499675


# # Ex 2 - The Two Envelopes Problem (20 Points)
# The Two Envelopes problem is an interesting decision theory problem.
# 
# > Consider that you are given two indistinguishable envelopes, each containing money, one contains twice as much as the other. You may pick one envelope and keep the money it contains. Having chosen an envelope at will, but before inspecting it, you are given the chance to switch envelopes. Should you switch?
# 
# > The game is this: "stick" or "switch"; It seems obvious that there is no point in switching envelopes as the situation is symmetric. However, because you stand to gain twice as much money if you switch while risking only a loss of half of what you currently have, it is possible to argue that it is more beneficial to switch. The problem is to show what is wrong with this argument.
# 
# **Two Envelopes problem**: Implement a function, called `simulateProblem()`, that does the game simulation for the two envelopes problem. Run the simulation 10000 times to figure out the empirical (observed) probability of gaining more money when switching and gaining more money when sticking to the original choice. Each simulation operates as follows:
# 1. First, randomly pick an envelopes configuration out of the two possible configurations, $(A,2A)$ or $(2A,A)$. In the first configuration, the second envelope has twice the money and in the second configuration, the first envelope has twice the money.
# 2. Next, randomly pick one of the two envelopes. 
# 3. Finally, randomly choose to either stick or switch. The program checks if you won (the envelope that picked has more money) or not (the envelope that picked has less money). In case of winning, record if the winning was because of _sticking_ or _switching_.
# 
# You can perform the _random_ choice as follows, using the `np.random.randint()` method.
# ```python
# import numpy as np
# print(np.random.randint(2))
# ```
# The `simulateProblem()` function takes no arguments and returns two values, first is a boolean output which is `True` if you win and `False` if you lose. In case of a win, the second output is `True` if the win was due to _sticking_ or the lose was due to _switching_ and `False` if the win was due to _switching_ or the lose was due to _sticking_.
# 
# Once the method `simulateProblem()` that does the above steps and returns "sticking",or "switching", depending on the win/loss scenario, run the method 1000 times and count the number of times the win was due to _sticking_ to the pick in Step 2, and number of times the win was due to _switching_ the envelope.

# In[4]:


import numpy as np

def simulateProblem():
    # YOUR CODE HERE
    
    # Step 1, Random pick an envelopes configureation out of the two possible configurations
    ## Inital Envelope config
    env = []
    
    if np.random.randint(2):
        env = [1,2]
    else:
        env = [2,1]   
    
    # Step 2, randomly pick one of the two envelope
    pick = np.random.randint(2)
    
    # Step 3, randomly chose "stick" or "switch"
    stick = np.random.randint(2)
    
    ## returns two values, first is a boolean output which is True if you win and False if you lose.
    ## In case of a win, the second output is True if the win was due to sticking or the lose was due to switching 
    ## and False if the win was due to switching or the lose was due to sticking.
    if stick:
        if env[pick] == 2:
            ### Win
            return [True, True]
        return [False, True]
    else:
        if env[pick] == 2:
            ### Lose
            return [False, False]
        return [True, False]


# In[5]:


assert type(simulateProblem()[0]) == bool
assert type(simulateProblem()[1]) == bool


# In[6]:


def run_simulation():
    """
    The function Run the simulation 10000 times to figure out 
    the empirical (observed) probability of gaining more money when switching 
    and gaining more money when sticking to the original choice.
    Return the probability of win due to sticking and win due to switching
    """
    # YOUR CODE HERE
    
    # Init
    win_sticking = 0
    win_switching = 0
    
    # Run the simulation 10000 times
    for time in range(0,10000):
        result = simulateProblem()
        # If game is win due to sticking
        if result[0] and result[1]:
            win_sticking += 1
        # If game is win due to switching
        elif result[0] and not result[1]:
            win_switching += 1

    return [win_sticking/10000,win_switching/10000]


# In[7]:



win_due_to_sticking, win_due_to_switching = run_simulation()
assert(abs(win_due_to_sticking-0.25) < 0.05)
assert(abs(win_due_to_switching-0.25) < 0.05)


# In[8]:


# EX3

class IteratorClass:
    ## Complete this class! It takes in three inputs when initializing. 
    # input#1 x -- is a sequence, either a list or a tuple. Raise a ValueError if it is not a list or a tuple
    # input#2 y -- is a sequence, either a list or a tuple. Raise a ValueError if it is not a list or a tuple
    # input#3 operator -- is a string that can either be 'add', 'sub', 'mul', 'div' -- If the specified operator
    # is not one of these, raise a ValueError. 
    
    # Complete the class by writing functions that will turn it into an iterator class. 
    # https://www.programiz.com/python-programming/methods/built-in/iter
    # The purpose of the class is to take two lists(x and y), apply the specified operator and return the output
    # as an iterator, meaning you can do "for ele in IteratorClass(x,y,'add')"
    
    
    # YOUR CODE HERE
    def __init__(self, x, y, operator):
        
        # Check the type of x, y, operator
        if type(x) != list and type(x) != tuple:
            raise ValueError('x should be list or tuple')
        if type(y) != list and type(y) != tuple:
            raise ValueError('y should be list or tuple')
            
        if type(operator) != str:
            raise ValueError('Operator should be string')
        else:
            if operator != 'add' and operator != 'sub' and operator != 'mul' and operator != 'div':
                raise ValueError("Operator should be either be 'add', 'sub', 'mul', 'div'")
                
        self.x = x
        self.y = y
        self.max = len(x)
        self.index = -1
        self.operator = operator
    
    # Iterator part
    def __iter__(self):
        return self
        
    def __next__(self):
        self.index += 1
        if self.index >= self.max:
            raise StopIteration
        
        # Return the result after implying operator
        if self.operator == 'add':
            return self.x[self.index] + self.y[self.index]
        elif self.operator == 'sub':
            return self.x[self.index] - self.y[self.index]
        elif self.operator == 'mul':
            return self.x[self.index] * self.y[self.index]
        elif self.operator == 'div':
            return round(self.x[self.index] / self.y[self.index],2)


# In[9]:


from nose.tools import assert_raises
assert_raises(ValueError, IteratorClass, range(5,10), [1,2,3], 'div')
assert_raises(ValueError, IteratorClass, [1,2,3], range(5,10), 'div')
assert_raises(ValueError, IteratorClass, [1,2,3], [1,2,3], 'abc')


# In[10]:


x = list(range(5,10))
y = list(range(50,55))

add_iterator = IteratorClass(x,y,'add')
sub_iterator = IteratorClass(x,y,'sub')
mul_iterator = IteratorClass(x,y,'mul')
div_iterator = IteratorClass(x,y,'div')

assert [ele for ele in add_iterator] == [55, 57, 59, 61, 63]
assert [ele for ele in sub_iterator] == [-45, -45, -45, -45, -45]
assert [ele for ele in mul_iterator] == [250, 306, 364, 424, 486]
assert [ele for ele in div_iterator] == [0.1, 0.12, 0.13, 0.15, 0.17]


# In[ ]:




