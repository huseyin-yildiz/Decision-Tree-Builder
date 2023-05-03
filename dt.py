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
        
        self.decision_value = None
        self.data_type = None

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
        data_type = self.attribute_types[split_pt[0]]
        decision_value = split_pt[1]
        
        # Numeric
        if(data_type == 1):
            mask = x[:,split_pt[0]] < decision_value
            
        # Categoric
        elif(data_type == 2):
            mask = x[:,split_pt[0]] == decision_value
        
        # Invalid attribute type
        else:
            raise TypeError("Attribute type is not valid. Only 1 and 2 is valid ")

        left = Node(x[mask], y[mask], attribute_types)
        right = Node(x[~mask], y[~mask], attribute_types)

        return (left, right)
    

    # Calculate the decision score by using left and right nodes. 
    # Explanation: Calculate the entropies of both left and right nodes and normalize them. 
    def calc_decision_score(self, node_left:'Node', node_right:'Node'):
        
        # Sizes of nodes
        left_size = float( len(node_left.x) )
        right_size = float( len(node_right.x) )
        total = left_size + right_size

        # Normalization
        score = left_size / total * left.calc_entropy() + right_size / total * right.calc_entropy()

        return score





# Recursive and entropy based DT generation algorithm  by using greedy algorithm
def generate_tree(node:Node , max_depth):

    # If max depth is reached, then terminates
    if(max_depth <= 0):
        return None
    
    # If the entropy of the current node is 0 then no need to continue anymore
    if(node.calc_entropy() == 0):
        return None
    
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

    
    # Best split found.






########## TESTS ##########




import pandas as pd
import numpy as np

# df = pd.read_csv("trial.csv")


# create a pandas DataFrame
df = pd.DataFrame({
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'Age': [25, 42, 32, 18, 47],
    'Income': [50000, 60000, 70000, 45000, 80000],
    'MaritalStatus': ['Married', 'Single', 'Married', 'Single', 'Married'],
    'BoughtInsurance': ['Yes', 'No', 'Yes', 'No', 'Yes']
})

print(df,"\n")

# Split x and y
y = df["BoughtInsurance"].values
x = df.drop("BoughtInsurance",axis=1)

attribute_types = [2,1,1,2]
x = x.values


root = Node(x,y,attribute_types)

val = root.calc_entropy()
print("root entropy:",val,"\n")


split_pts = root.find_split_pts()

print("Split Points:")
for pt in split_pts:
    left,right = root.generate_nodes(pt)
    split_score = root.calc_decision_score(left,right)
    print(pt, "Score:", split_score)
    
