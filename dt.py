import numpy as np 
import math



# Extract the midpoints from given feature column
def get_midpoints(feature_data):
    # Sort the data in ascending order
    sorted_data = sorted(feature_data)
    
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
        self.classes = np.unique(y)
        self.class_label = None     # If the leaf node

        self.decision_value = None
        self.decision_type = None
        self.decision_attribute = None

    # Calculate the entropy of the node with their data
    def calc_entropy(self):
        entropy = 0
        for cls in self.classes:
            probab = np.count_nonzero(self.y == cls) / self.y.size
            entropy -= probab * math.log2(probab)
        return entropy

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
                split_pts +=  [ (i,x) for x in uniques ]
            
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

        left = Node(self.x[mask], self.y[mask], attribute_types)
        right = Node(self.x[~mask], self.y[~mask], attribute_types)

        return (left, right)
    

    # Calculate the decision score by using left and right nodes. 
    # Explanation: Calculate the entropies of both left and right nodes and normalize them. 
    def calc_decision_score(self, node_left:'Node', node_right:'Node'):
        
        # Sizes of nodes
        left_size = float( len(node_left.x) )
        right_size = float( len(node_right.x) )
        total = left_size + right_size

        # Normalization
        score = left_size / total * node_left.calc_entropy() + right_size / total * node_right.calc_entropy()

        return score


    # Predict the given data
    # Go until leaf node recursively and return class label
    def predict(self, data):
        
        if(self.class_label != None):
            return self.class_label

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
def generate_tree(node:Node , max_depth):

    # If max depth is reached, then label the leaf node and terminates
    if(max_depth <= 0):
        counts = np.bincount(node.y)
        most_freq = np.argmax(counts)
        node.class_label = node.y[most_freq]
        return
    
    # If the entropy of the current node is 0 then no need to continue anymore
    # Label the leaf node and return
    if(node.calc_entropy() == 0):
        node.class_label = node.y[0]
        return
    
    # Find all split points
    split_points = node.find_split_pts()

    # Variables to hold best split
    best_split = None
    best_score = 1.1
    nodes = None

    # Calculate scores of all split points and get the best split
    for split_pt in split_points:
        
        # Generate the child nodes with split
        node_left, node_right = node.generate_nodes(split_pt)
        
        # Calculate score
        score = node.calc_decision_score(node_left, node_right)
        
        if( score < best_score ):
            best_score = score
            best_split = split_pt
            nodes = (node_left, node_right)

    
    # End of For : Best split found.

    # Place the children to the left and right 
    node.left, node.right = node_left, node_right
    
    # Place the decision value and data type on the node
    node.decision_value = best_split[1]
    node.decision_type = node.attribute_types[best_split[0]]
    node.decision_attribute = best_split[0]

    # Recursive call for children
    generate_tree(node.left, max_depth-1)
    generate_tree(node.right, max_depth-1)
    



# DT builder
def buid_dt(X, y, attribute_types, max_depth):
    root = Node(X,y,attribute_types)
    generate_tree(root, max_depth)
    return root





# Takes DT and X matrix returns a vector for predicted predicted labels
def predict_dt(dt:Node, X):
    predict_vector = [dt.predict(x) for x in X]
    return np.array( predict_vector )



########## TESTS ##########




import pandas as pd
import numpy as np


# # create a pandas DataFrame
# df = pd.DataFrame({
#     'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
#     'Age': [25, 42, 32, 18, 47],
#     'Income': [50000, 60000, 70000, 45000, 80000],
#     'MaritalStatus': ['Married', 'Single', 'Married', 'Single', 'Married'],
#     'BoughtInsurance': ['Yes', 'No', 'Yes', 'No', 'Yes']
# })

# print(df,"\n")

# # Split x and y
# y = df["BoughtInsurance"].values
# x = df.drop("BoughtInsurance",axis=1)

# attribute_types = [2,1,1,2]
# x = x.values


# root = Node(x,y,attribute_types)

# val = root.calc_entropy()
# print("root entropy:",val,"\n")


# split_pts = root.find_split_pts()

# print("Split Points:")
# for pt in split_pts:
#     left,right = root.generate_nodes(pt)
#     split_score = root.calc_decision_score(left,right)
#     print(pt, "Score:", split_score)
    

# # generate_tree(root,5)


# dt = buid_dt(x,y,attribute_types,5)
# print("ok")

# for i,data in enumerate(x):
#     predicted = dt.predict(data)
#     if(predicted == y[i]):
#         print("True")
#     else:
#         print("False")
    
# predicted = predict_dt(dt,x)

# print("Predicted:",predicted)
# print("Actual   :",y)





# Test on huge dataset



df1 = pd.read_csv("trial.csv")

# Split x and y
Y = df1["Risk"].values
X = df1.drop("Risk",axis=1)
X = df1.drop("LOCATION_ID",axis=1).values


attribute_types = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report

k_fold = KFold(n_splits=6, shuffle=True, random_state=42)

print(X[1])

for k, (train, test) in enumerate(k_fold.split(X, Y)):
  # Train
  dt = buid_dt(X[train], Y[train],attribute_types,5)
  y_pred = predict_dt(dt, X[test])
  print("\n\nFold",k,":")
  # confusion matrix
  conf_mat = confusion_matrix(Y[test], y_pred)
  # Display confusion matrix
  cm_df = pd.DataFrame(conf_mat, columns=['Predicted 0', 'Predicted 1'], index=['True 0', 'True 1'])
  print('Confusion matrix:')
  print(cm_df)
  print("\nResult:")
  print( classification_report(Y[test],y_pred) )
  print()



