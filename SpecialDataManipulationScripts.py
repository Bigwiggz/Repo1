############################################################################## 
#Data Manipulation
##############################################################################



#FUNCTION-1: Multiple comma seperated values that need to be split spread accross pandas column
##############################################################################

#This function is to be used with any column that has multiple comma seperated values that need to be split
#This function is to be used in conjuction with .apply(lambda x:x)
import csv


#Create df_Result
column_names = ["Auto ID", "PhoneNumber"]
df_Result=pd.DataFrame(columns=column_names)
data=[]
def readInCSVColumn(ColumnIndex,ColumnToSplit):
  reader = csv.reader(ColumnToSplit.split(','), delimiter=',')
  for row in reader:
    Phone_Number='\t'.join(row)
    values=[ColumnIndex,Phone_Number]
    zipped = zip(column_names , values)
    new_row = {column_names[0]:ColumnIndex, column_names[1]:Phone_Number}
    print(new_row)
    a_dictionary = dict(zipped)
    data.append(a_dictionary)
	
#Run Function
df2.apply(lambda x: readInCSVColumn(x['Auto ID'],x['Phone_y']), axis=1)
print(data)

# Save data to a dataframe
df_Result=df_Result.append(data, True)


#FUNCTION-2: concatenate multiple items into a csv per field per index
##############################################################################

#This function is the reverse of function 1.  It combines multiple items into a csv per field per index
#This function is to be used in conjuction with .apply(lambda x:x)
#CONCATENATE ALL NAMES

#set indexing dataframe
df_Name=df

#set Address index
df_Name.set_index(["Auto ID"])

# filter unncessary columns
df_Name=df_Name.filter(["Auto ID","Full_Name"])
df_Name.drop_duplicates(keep="first",inplace=True)

#Group-by all records on that index of Phone and Physical Address
df_Name=df_Name.groupby(["Auto ID"],sort=False)["Full_Name"].apply(lambda x: ', '.join(map(str, x)))

#Add Names List back into original dataframe by using a left outer join
df=pd.merge(df,df_Name,left_on=["Auto ID"],
           right_on=["Auto ID"],
           how='left')


