import numpy as np 
import math



# Extract the midpoints from given feature column
def get_midpoints(feature_data):
    # Sort the data in ascending order
    sorted_data = sorted(feature_data)
    # Remove duplicates
    sorted_data = np.unique(sorted_data)
    
    # Find midpoints between adjacent values
    midpoints = []
    for i in range(len(sorted_data) - 1):
        midpoint = (sorted_data[i] + sorted_data[i+1]) / 2.0
        midpoints.append( midpoint )
        
    return midpoints


# Class for decision nodes 
class Node:
    
    # Init function 
    def __init__(self, x, y, attribute_types):
        self.x = x
        self.y = y
        self.attribute_types = attribute_types
        self.left = None
        self.right = None
        self.regress_value = None     # If the leaf node

        self.decision_value = None
        self.decision_type = None
        self.decision_attribute = None

    # Calculates the error on a node
    def calc_error(self):
        mean = np.mean(self.y)
        sum_err = 0    
        for r in self.y:
            sum_err += (r - mean)**2
        error = sum_err / len(self.y)
        return error

    # Find all split points on the data
    def find_split_pts(self):
        split_pts = list()
        
        # Traverse all attributes
        for (i,attribute) in enumerate(self.attribute_types):
            
            # Numeric   
            if(attribute == 1):               
                mid_pts = get_midpoints(self.x[:,i])
                split_pts += [(i,x) for x in mid_pts]

            # Categoric
            elif(attribute == 2):             
                uniques = np.unique(self.x[:,i])
                
                # Binary class --> just add one of them
                if(len(uniques) == 2):
                    split_pts.append( (i, uniques[0]) )
                
                # Multiclass --> add each one
                elif(len(uniques) > 2):
                    split_pts +=  [ (i,x) for x in uniques ]
                
                # If It has only one unique class for this data then do not add it to split points

            # Invalid attribute type
            else:
                raise TypeError("Attribute type is not valid. Only 1 and 2 is valid ")

        return  split_pts


    # Generate left and right nodes with given split point
    def generate_nodes(self, split_pt):
        decision_type = self.attribute_types[split_pt[0]]
        decision_value = split_pt[1]
        
        # Numeric
        if(decision_type == 1):
            mask = self.x[:,split_pt[0]] < decision_value
            
        # Categoric
        elif(decision_type == 2):
            mask = self.x[:,split_pt[0]] == decision_value
        
        # Invalid attribute type
        else:
            raise TypeError("Attribute type is not valid. Only 1 and 2 is valid ")

        
        if(len(self.x[mask]) == 0):
            raise "split error"
        if(len(self.x[~mask]) == 0):
            raise "split error"
        

        left = Node(self.x[mask], self.y[mask], self.attribute_types)
        right = Node(self.x[~mask], self.y[~mask], self.attribute_types)

        return (left, right)
    

    # Calculate the split error by using left and right nodes. 
    # Explanation: Calculate the error of both left and right nodes and normalize them. 
    def calc_split_error(self, node_left:'Node', node_right:'Node'):
        
        # Sizes of nodes
        left_size = float( len(node_left.x) )
        right_size = float( len(node_right.x) )
        total = left_size + right_size

        # Normalization
        split_error = left_size / total * node_left.calc_error() + right_size / total * node_right.calc_error()

        return split_error


    # Predict the given data
    # Go until leaf node recursively and return regress value
    def predict(self, data):
        
        if(self.regress_value != None):
            return self.regress_value

        # Numeric
        if(self.decision_type == 1):
           if( data[self.decision_attribute] < self.decision_value):
               return self.left.predict(data)
           
           else:
               return self.right.predict(data)    
        
        # Categoric
        elif(self.decision_type == 2):
            if( data[self.decision_attribute] == self.decision_value):
               return self.left.predict(data)
           
            else:
                return self.right.predict(data)    
        

        # Invalid attribute type
        else:
            raise TypeError("Attribute type is not valid. Only 1 and 2 is valid ")



# Recursive and entropy based DT generation algorithm  by using greedy algorithm
# Take data included root node and generate the DT recursively
# Arg "error_limit" : limits the DT with a error rate. If reach this error rate then terminate the node 
def generate_tree(node:Node , max_depth, error_limit):

    # If max depth is reached, then label the leaf node with mean value and return
    if(max_depth <= 0):
        node.regress_value = np.mean(node.y)
        return
    
    # If the error of the current node is 0 then no need to continue anymore
    # Label the leaf node with mean value and return
    if(node.calc_error() <= error_limit):
        node.regress_value = np.mean(node.y)
        return
    
    # Find all split points
    split_points = node.find_split_pts()

    # Variables to hold best split
    best_split = None
    least_error = float('inf')
    nodes = None

    # Calculate errors of all split points and get the best split
    for split_pt in split_points:
        
        # Generate the child nodes with split
        node_left, node_right = node.generate_nodes(split_pt)
        
        # Calculate error
        error = node.calc_split_error(node_left, node_right)
        
        if( error < least_error ):
            least_error = error
            best_split = split_pt
            nodes = (node_left, node_right)

    
    # End of For : Best split found.

    # Place the children to the left and right 
    node.left, node.right = nodes[0], nodes[1]
    
    # Place the decision value and data type on the node
    node.decision_value = best_split[1]
    node.decision_type = node.attribute_types[best_split[0]]
    node.decision_attribute = best_split[0]

    # Recursive call for children
    generate_tree(node.left, max_depth-1, error_limit)
    generate_tree(node.right, max_depth-1, error_limit)
    



# DT regressor builder
def buid_rdf(X, y, attribute_types, max_depth, error_limit):
    root = Node(X,y,attribute_types)
    generate_tree(root, max_depth, error_limit)
    return root



# Takes DT and X matrix returns a vector for predicted predicted labels
def predict_rdf(dt:Node, X):
    predict_vector = [dt.predict(x) for x in X]
    return np.array( predict_vector )



########## TESTS ##########

import pandas as pd

# create a pandas DataFrame
# df = pd.DataFrame({
#     'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
#     'Age': [25, 42, 32, 18, 47],
#     'MaritalStatus': ['Married', 'Single', 'Married', 'Single', 'Married'],
#     'BoughtInsurance': ['Yes', 'No', 'Yes', 'No', 'Yes'],
#     'Income': [50000, 60000, 70000, 45000, 80000]
# })

# print(df,"\n")

# # Split x and y
# y = df["Income"].values
# x = df.drop("Income",axis=1)

# attribute_types = [2,1,2,2]
# x = x.values


# root = Node(x,y,attribute_types)

# val = root.calc_error()
# print("root error:",val,"\n")



# split_pts = root.find_split_pts()

# print("Split Points:")
# for pt in split_pts:
#     left,right = root.generate_nodes(pt)
#     split_score = root.calc_split_error(left,right)
#     print(pt, "Split error:", split_score)
    

# generate_tree(root,5)




# dt = buid_rdf(x,y,attribute_types,max_depth=5, error_limit=0)
# print("ok")

# for i,data in enumerate(x):
#     predicted = dt.predict(data)
#     if(predicted == y[i]):
#         print("True")
#     else:
#         print("False")
    
# predicted = predict_rdf(dt,x)

# print("Predicted:",predicted)
# print("Actual   :",y)







# Test on huge dataset



df1 = pd.read_csv("day.csv")


# Split x and y
Y = df1["cnt"].values
X = df1.drop("cnt",axis=1)
X = X.drop("dteday",axis=1)
X = X.drop("instant",axis=1)


print(X.info())

X = X.values

attribute_types = [2,2,2,2,2,2,2,1,1,1,1,1,1]

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, r2_score


k_fold = KFold(n_splits=6, shuffle=True, random_state=42)


mean=0    

for k, (train, test) in enumerate(k_fold.split(X, Y)):
  # Train
  dt = buid_rdf(X[train], Y[train], attribute_types, max_depth=5, error_limit=0)
  y_pred = predict_rdf(dt, X[test])
  print("\nResult:")
  # Evaluate the model using R^2 score
  r2 = r2_score(Y[test], y_pred)
  print("R^2 score: {:.2f}".format(r2))
  mean += r2

mean = mean / 6
print("Mean R^2 score:", mean)


