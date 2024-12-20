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


from IPython.display import display, HTML
import pandas as pd
import sqlite3
from sqlite3 import Error

def create_connection(db_file, delete_db=False):
    import os
    if delete_db and os.path.exists(db_file):
        os.remove(db_file)

    conn = None
    try:
        conn = sqlite3.connect(db_file)
        conn.execute("PRAGMA foreign_keys = 1")
    except Error as e:
        print(e)

    return conn


def create_table(conn, create_table_sql):
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)
        
def execute_sql_statement(sql_statement, conn):
    cur = conn.cursor()
    cur.execute(sql_statement)

    rows = cur.fetchall()

    return rows


# In[3]:


conn = create_connection('non_normalized.db')
sql_statement = "select * from Students;"
df = pd.read_sql_query(sql_statement, conn)
display(df)


# In[4]:


def normalize_database(non_normalized_db_filename):
#     Normalize 'non_normalized.db'
#     Call the normalized database 'normalized.db'
#     Function Output: No outputs
#     Requirements:
#     Create four tables
#     Degrees table has one column:
#         [Degree] column is the primary key
    
#     Exams table has two columns:
#         [Exam] column is the primary key column
#         [Year] column stores the exam year
    
#     Students table has four columns:
#         [StudentID] primary key column 
#         [First_Name] stores first name
#         [Last_Name] stores last name
#         [Degree] foreign key to Degrees table
        
#     StudentExamScores table has four columns:
#         [PK] primary key column,
#         [StudentID] foreign key to Students table,
#         [Exam] foreign key to Exams table ,
#         [Score] exam score


    # YOUR CODE HERE
    conn = create_connection(non_normalized_db_filename)
    
    import re
    
    with conn:
        val_list = []
        
        # Get degree info
        sql_statement_degree = """SELECT Degree FROM Students group by Degree ;"""
        rows = execute_sql_statement(sql_statement_degree, conn)
        deg_list = []
        for row in rows:
            deg_list.append(row[0])
        val_list.append(deg_list)
        
        # Get exams info unique
        sql_statement_exam = """SELECT Exams FROM Students;"""
        rows = execute_sql_statement(sql_statement_exam, conn)
        exam_list = []
        for row in rows:
            for exam in row[0].split(','):
                exam = re.sub(' |\)','',exam)
                exam = re.sub('\(',',',exam)
                if exam not in exam_list: 
                    exam_list.append(exam)
        
                
        val_list.append(exam_list)
        
        # Get student indo
        sql_statement_stu = """SELECT StudentID, substr(NAME, pos+1) AS First_Name, 
        substr(NAME, 1, pos-1) AS Last_Name, Degree
        FROM (SELECT *, instr(NAME,', ') AS pos FROM Students)"""
        stu_list = []
        rows = execute_sql_statement(sql_statement_stu, conn)
        for row in rows:
            stu_list.append([row[0], row[1].strip(), row[2].strip(), row[3]])
        val_list.append(stu_list)
        
        # Get student indo
        sql_statement_ses = """SELECT StudentID, Exams, Scores FROM Students ;"""
        rows = execute_sql_statement(sql_statement_ses, conn)
        ses_list = []
        ses_num = 1
        for row in rows:
            for exam_info, score in zip(row[1].split(','),row[2].split(',')):
                exam_info = re.sub(' ','',exam_info)
                exam = exam_info.split('(')
                ses_list.append([ses_num, row[0], exam[0], score])
                ses_num += 1
        val_list.append(ses_list)

    
    conn = create_connection('normalized.db')
    
    with conn:
        # Create table
        create_tables = []
        
        # Delete table
        #execute_sql_statement("DROP TABLE StudentExamScores", conn)
        #execute_sql_statement("DROP TABLE Students", conn)
        #execute_sql_statement("DROP TABLE Exams", conn)
        #execute_sql_statement("DROP TABLE Degrees", conn)
        
        # create table list
        create_tables.append(""" CREATE TABLE [Degrees] (
        [Degree] TEXT  NOT NULL PRIMARY KEY
        ); """)
        
        create_tables.append(""" CREATE TABLE [Exams] (
        [Exam] TEXT  NOT NULL PRIMARY KEY,
        [Year] INTEGER  NULL
        ); """)
        
        create_tables.append(""" CREATE TABLE [Students] (
        [StudentID]  INTEGER  NOT NULL PRIMARY KEY,
        [First_Name] TEXT NULL,
        [Last_Name]  TEXT NULL,
        [Degree]     TEXT NULL,
        FOREIGN KEY(Degree) REFERENCES Degrees(Degree)
        ); """)
        
        create_tables.append(""" CREATE TABLE [StudentExamScores] (
        [PK]  INTEGER  NOT NULL PRIMARY KEY,
        [StudentID] INTEGER NULL,
        [Exam]      TEXT NULL,
        [Score]     INTEGER NULL,
        FOREIGN KEY(StudentID) REFERENCES Students(StudentID),
        FOREIGN KEY(Exam) REFERENCES Exams(Exam)
        ); """)       
        
        for table in create_tables:
            create_table(conn, table)
        
        # Insert data to table
        statement_list = []
        
        statement_list.append(''' INSERT INTO Degrees(Degree)
        VALUES(?) ''')
        statement_list.append(''' INSERT INTO Exams(Exam, Year)
        VALUES(?,?) ''')
        statement_list.append(''' INSERT INTO Students(StudentID, First_Name, Last_Name, Degree)
        VALUES(?,?,?,?) ''')
        statement_list.append(''' INSERT INTO StudentExamScores(PK, StudentID, Exam, Score)
        VALUES(?,?,?,?) ''')
        
        cur = conn.cursor()
        
        for statement,table_val in zip(statement_list,val_list):
            for values in table_val:
            
                if type(values) == str:
                    values = values.split(',')
                elif type(values) == tuple:
                    values = list(values)
                cur.execute(statement, values)
                
        #Print table
        #sql_statement = "SELECT * FROM Degrees;"
        #df = pd.read_sql_query(sql_statement, conn)
        #print(df)
        


# In[5]:


normalize_database('non_normalized.db')
conn = create_connection('normalized.db')


# In[6]:


def ex1():
    # Write an SQL statement that SELECTs all rows from the `Exams` table and sort the exams by Year
    # output columns: exam, year
    
    # YOUR CODE HERE
    sql_statement = "select exam, year from Exams order by year,exam"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[7]:


sql_statement = ex1()
data = pd.read_csv("ex1.csv")
cur = conn.cursor()
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[8]:


def ex2():
    # Write an SQL statement that SELECTs all rows from the `Degrees` table and sort the degrees by name
    # output columns: degree
    
    # YOUR CODE HERE
    sql_statement = "select Degree from Degrees order by Degree"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[9]:


sql_statement = ex2()
data = pd.read_csv("ex2.csv")
cur = conn.cursor()
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[10]:


def ex3():
    # Write an SQL statement that counts the numbers of gradate and undergraduate students
    # output columns: degree, count_degree
    
    # YOUR CODE HERE
    sql_statement = "select degree, count(degree) as count_degree from Students group by degree"
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[11]:


sql_statement = ex3()
data = pd.read_csv("ex3.csv")
cur = conn.cursor()
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[12]:


def ex4():
    # Write an SQL statement that calculates the exam averages for exams; sort by average in descending order.
    # output columns: exam, year, average
    # round to two decimal places
    
    
    # YOUR CODE HERE
    sql_statement = """select Exams.exam, Exams.year, round(avg(StudentExamScores.score),2) as average from StudentExamScores 
    inner join Exams on Exams.exam = StudentExamScores.exam
    group by Exams.exam
    order by average DESC"""
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[13]:


sql_statement = ex4()
data = pd.read_csv("ex4.csv")
cur = conn.cursor()
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[14]:


def ex5():
    # Write an SQL statement that calculates the exam averages for degrees; sort by average in descending order.
    # output columns: degree, average 
    # round to two decimal places
    
    # YOUR CODE HERE
    sql_statement = """select Degrees.degree as Degree, round(avg(StudentExamScores.score),2) as average 
    from StudentExamScores
    inner join Students on Students.StudentID = StudentExamScores.StudentID
    inner join Exams on Exams.exam = StudentExamScores.exam
    inner join Degrees on Degrees.degree = Students.degree
    group by Degrees.degree
    order by average DESC
    """
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[15]:


sql_statement = ex5()
data = pd.read_csv("ex5.csv")
cur = conn.cursor()
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[16]:


def ex6():
    # Write an SQL statement that calculates the exam averages for students; sort by average in descending order. Show only top 10 students
    # output columns: first_name, last_name, degree, average
    # round to two decimal places
    
    # YOUR CODE HERE
    sql_statement = """ select Students.First_Name, Students.Last_Name, Degrees.degree as Degree, 
    round(avg(StudentExamScores.score),2) as average
    from StudentExamScores
    inner join Students on Students.StudentID = StudentExamScores.StudentID
    inner join Degrees on Degrees.degree = Students.degree
    group by Students.StudentID
    order by average DESC limit 10
    """
    df = pd.read_sql_query(sql_statement, conn)
    display(df)
    return sql_statement


# In[17]:


sql_statement = ex6()
data = pd.read_csv("ex6.csv")
cur = conn.cursor()
df = pd.read_sql_query(sql_statement, conn)
assert df.equals(data) == True


# In[ ]:




