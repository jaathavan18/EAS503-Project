#!/usr/bin/env python
# coding: utf-8

# Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).
# 
# Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

# In[1]:


NAME = "chiehjul"
COLLABORATORS = ""


# ---

# # Instructions 
# For every exercise, simply write your sql statement in a variable called `sql_statement`. `output columns` tells you which columns to include in your select statement. 

# In[2]:


# Setup
from IPython.display import display, HTML
import pandas as pd
import sqlite3
conn = sqlite3.connect("homework6.db")
cur = conn.cursor()


# In[3]:


sql_statement = "select * from customers;"
df = pd.read_sql_query(sql_statement, conn)
display(df)


# In[4]:


sql_statement = "select * from orders;"
df = pd.read_sql_query(sql_statement, conn)
display(df)


# In[5]:


sql_statement = "select * from vendors;"
df = pd.read_sql_query(sql_statement, conn)
display(df)


# In[6]:


sql_statement = "select * from products;"
df = pd.read_sql_query(sql_statement, conn)
display(df)


# In[7]:


sql_statement = "select * from orderitems;"
df = pd.read_sql_query(sql_statement, conn)
display(df)


# In[8]:


sql_statement = "select * from productnotes;"
df = pd.read_sql_query(sql_statement, conn)
display(df)


# In[9]:


def ex1():
    # Write an SQL statement that SELECTs all rows from the `customers` table
    # output columns: cust_name, cust_email

    # YOUR CODE HERE
    sql_statement = "select cust_name, cust_email from customers;"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[10]:


sql_statement = ex1()
data = pd.read_csv("ex1.csv") 
cur = conn.cursor()
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[11]:


def ex2():
    # Write an SQL statement that SELECTs all rows from the `products` table
    # output columns: vend_id

    # YOUR CODE HERE
    sql_statement = "select vend_id from products"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[12]:


sql_statement = ex2()
data = pd.read_csv("ex2.csv")
cur = conn.cursor()
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[13]:


def ex3():
    # Write an SQL statement that SELECTs distinct rows for vend_id from the `products` table
    # output columns: vend_id

    # YOUR CODE HERE
    sql_statement = "select vend_id from products group by vend_id;"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[14]:


sql_statement = ex3()
data = pd.read_csv("ex3.csv")
cur = conn.cursor()
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[15]:


def ex4():
    # Write an SQL statement that SELECTs the first five rows from the `products` table
    # output columns: prod_name

    # YOUR CODE HERE
    sql_statement = "select prod_name from products limit 5;"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[16]:


sql_statement = ex4()
data = pd.read_csv("ex4.csv")
cur = conn.cursor()
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[17]:


def ex5():
    # Write an SQL statement that SELECTs 4 rows starting from row 3 from `products` table
    # output columns: prod_name

    # YOUR CODE HERE
    sql_statement = "select prod_name from products limit 4 offset 3"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[18]:


sql_statement = ex5()
data = pd.read_csv("ex5.csv")
cur = conn.cursor()
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[19]:


def ex6():
    # Write an SQL statement that SELECTs all rows from `products` table and sorts by prod_name
    # output columns: prod_name

    # YOUR CODE HERE
    sql_statement = "select prod_name from products order by prod_name"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[20]:


sql_statement = ex6()
data = pd.read_csv("ex6.csv")
cur = conn.cursor()
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[21]:


def ex7():
    # Write an SQL statement that SELECTs all rows from `products` table and sorts by prod_price and then prod_name 
    # output columns: prod_id, prod_price, prod_name

    # YOUR CODE HERE
    sql_statement = "select prod_id, prod_price, prod_name from products order by prod_price, prod_name"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[22]:


sql_statement = ex7()
data = pd.read_csv("ex7.csv")
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[23]:


def ex8():
    # Write an SQL statement that SELECTs all rows from `products` table and sorts by prod_price (descending order)
    # and then prod_name 
    # output columns: prod_id, prod_price, prod_name

    # YOUR CODE HERE
    sql_statement = "select prod_id, prod_price, prod_name from products order by prod_price desc, prod_name"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[24]:


sql_statement = ex8()
data = pd.read_csv("ex8.csv")
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[25]:


def ex9():
    # Write an SQL statement that SELECTs all rows from `products` table where the price of product is 2.50
    # output columns: prod_id, prod_price, prod_name

    # YOUR CODE HERE
    sql_statement = "select prod_id, prod_price, prod_name from products where prod_price is 2.50"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[26]:


sql_statement = ex9()
data = pd.read_csv("ex9.csv")
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[27]:


def ex10():
    # Write an SQL statement that SELECTs all rows from `products` table where the name of product is Oil can
    # output columns: prod_id, prod_price, prod_name

    # YOUR CODE HERE
    sql_statement = "select prod_id, prod_price, prod_name from products where prod_name like 'Oil can'"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[28]:


sql_statement = ex10()
data = pd.read_csv("ex10.csv")
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[29]:


def ex11():
    # Write an SQL statement that SELECTs all rows from `products` table where the price of product is 
    # less than or equal to 10
    # output columns: prod_id, prod_price, prod_name

    # YOUR CODE HERE
    sql_statement = "select prod_id, prod_price, prod_name from products where prod_price <= 10"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[30]:


sql_statement = ex11()
data = pd.read_csv("ex11.csv")
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[31]:


def ex12():
    # Write an SQL statement that SELECTs all rows from `products` table where the vendor id is not equal to 1003
    # output columns: vend_id, prod_name

    # YOUR CODE HERE
    sql_statement = "select vend_id, prod_name from products where vend_id != 1003"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[32]:


sql_statement = ex12()
data = pd.read_csv("ex12.csv")
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[33]:


def ex13():
    # Write an SQL statement that SELECTs all rows from `products` table where the product prices are between 5 and 10
    # output columns: prod_name, prod_price

    # YOUR CODE HERE
    sql_statement = "select prod_name, prod_price from products where prod_price between 5 and 10"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[34]:


sql_statement = ex13()
data = pd.read_csv("ex13.csv")
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[35]:


def ex14():
    # Write an SQL statement that SELECTs all rows from the `customers` table where the customer email is empty
    # output columns: cust_id, cust_name

    # YOUR CODE HERE
    sql_statement = "select cust_id, cust_name from customers where cust_email is null"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[36]:


sql_statement = ex14()
data = pd.read_csv("ex14.csv")
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[37]:


def ex15():
    # Write an SQL statement that SELECTs all rows from the `products` table where the vender id is 1003 and
    # the price is less than or equal to 10. 
    # output columns: vend_id, prod_id, prod_price, prod_name

    # YOUR CODE HERE
    sql_statement = "select vend_id, prod_id, prod_price, prod_name from products where vend_id = 1003 and prod_price <= 10"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[38]:


sql_statement = ex15()
data = pd.read_csv("ex15.csv")
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[39]:


def ex16():
    # Write an SQL statement that SELECTs all rows from the `products` table where the vender id is 1002 or 1003 and
    # the price is greater than or equal to 5. 
    # output columns: vend_id, prod_id, prod_price, prod_name

    # YOUR CODE HERE
    sql_statement = "select vend_id, prod_id, prod_price, prod_name from products where (vend_id = 1002 or vend_id = 1003) and prod_price >= 5"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[40]:


sql_statement = ex16()
data = pd.read_csv("ex16.csv")
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[41]:


def ex17():
    # Write an SQL statement that SELECTs all rows from the `products` table where the vender id is 1002 or 1003 or 1005.
    # Use the IN operator for this!
    # output columns: vend_id, prod_id, prod_price, prod_name

    # YOUR CODE HERE
    sql_statement = "select vend_id, prod_id, prod_price, prod_name from products where vend_id in (1002, 1003, 1005)"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[42]:


sql_statement = ex17()
data = pd.read_csv("ex17.csv")
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[43]:


def ex18():
    # Write an SQL statement that SELECTs all rows from the `products` table where the vender id is NOT 1002 or 1003.
    # Use the NOT IN operator for this!
    # output columns: vend_id, prod_id, prod_price, prod_name

    # YOUR CODE HERE
    sql_statement = "select vend_id, prod_id, prod_price, prod_name from products where vend_id not in (1002,1003)"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[44]:


sql_statement = ex18()
data = pd.read_csv("ex18.csv")
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[45]:


def ex19():
    # Write an SQL statement that SELECTs all rows from the `products` table where the product name starts with 'jet'
    # output columns: prod_id, prod_price, prod_name

    # YOUR CODE HERE
    sql_statement = "select prod_id, prod_price, prod_name from products where prod_name like 'jet%'"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[46]:


sql_statement = ex19()
data = pd.read_csv("ex19.csv")
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[47]:


def ex20():
    # Write an SQL statement that SELECTs all rows from the `products` table where the product name ends with 'jet'
    # output columns: prod_id, prod_price, prod_name

    # YOUR CODE HERE
    sql_statement = "select prod_id, prod_price, prod_name from products where prod_name like '%anvil'"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[48]:


sql_statement = ex20()
data = pd.read_csv("ex20.csv")
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[49]:


def ex21():
    # Write an SQL statement that SELECTs all rows from the `vendors` table. Concatenate vendor name and vendor country
    # as vend_title. Leave space in between -- example `ACME (USA)`
    # output columns: vend_title

    # YOUR CODE HERE
    sql_statement = "select vend_name || ' (' || vend_country || ')' as vend_title from vendors order by vend_title"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[50]:


sql_statement = ex21()
data = pd.read_csv("ex21.csv")
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[51]:


def ex22():
    # Write an SQL statement that SELECTs all rows from the `orderitems` table where order number is 20005. 
    # Display an extra calculated column called `expanded_price` that is the result of quantity multiplied by item_price.
    # Round the value to two decimal places. 
    # output columns: prod_id, quantity, item_price, expanded_price

    # YOUR CODE HERE
    sql_statement = "select prod_id, quantity, item_price, round( quantity* item_price,2) as expanded_price from orderitems where order_num = 20005"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[52]:


sql_statement = ex22()
data = pd.read_csv("ex22.csv")
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[53]:


def ex23():
    # Write an SQL statement that SELECTs all rows from the `orders` table where the order date is between 
    # 2005-09-13 and 2005-10-04
    # output columns: order_num, order_date
    # https://www.sqlitetutorial.net/sqlite-date-functions/sqlite-date-function/
    
    # YOUR CODE HERE
    sql_statement = "select order_num, order_date from orders where date(order_date) between '2005-09-13' and '2005-10-04'"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[54]:


sql_statement = ex23()
data = pd.read_csv("ex23.csv")
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[55]:


def ex24():
    # Write an SQL statement that calculates the average price of all rows in the `products` table. 
    # Call the average column avg_price
    # output columns: avg_price

    # YOUR CODE HERE
    sql_statement = "select avg(prod_price) as avg_price from products"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[56]:


sql_statement = ex24()
data = pd.read_csv("ex24.csv")
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[57]:


def ex25():
    # Write an SQL statement that calculates the average price of all rows in the `products` table 
    # where the vendor id is 1003 . Call the average column avg_price
    # output columns: avg_price

    # YOUR CODE HERE
    sql_statement = "select avg(prod_price) as avg_price from products where vend_id = 1003"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[58]:


sql_statement = ex25()
data = pd.read_csv("ex25.csv")
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[59]:


def ex26():
    # Write an SQL statement that counts the number of customers in the `customers` table 
    # Call the count column num_cust
    # output columns: num_cust

    # YOUR CODE HERE
    sql_statement = "select count(*) as num_cust from customers"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[60]:


sql_statement = ex26()
data = pd.read_csv("ex26.csv")
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[61]:


def ex27():
    # Write an SQL statement that calculates the max price in the `products` table 
    # Call the max column max_price. Round the value to two decimal places. 
    # output columns: num_cust

    # YOUR CODE HERE
    sql_statement = "select max(prod_price) as max_price from products "
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[62]:


sql_statement = ex27()
data = pd.read_csv("ex27.csv")
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[63]:


def ex28():
    # Write an SQL statement that calculates the min price in the `products` table 
    # Call the min column min_price. Round the value to two decimal places. 
    # output columns: num_cust

    # YOUR CODE HERE
    sql_statement = "select min(prod_price) as min_price from products"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[64]:


sql_statement = ex28()
data = pd.read_csv("ex28.csv")
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[65]:


def ex29():
    # Write an SQL statement that sums the quantity in the `orderitems` table where order number is 20005. 
    # Call the sum column items_ordered
    # output columns: num_cust

    # YOUR CODE HERE
    sql_statement = "select sum(quantity) as items_ordered from orderitems where order_num = 20005"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[66]:


sql_statement = ex29()
data = pd.read_csv("ex29.csv")
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True

