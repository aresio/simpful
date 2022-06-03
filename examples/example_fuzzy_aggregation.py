from simpful import FuzzySet, FuzzyAggregator
from numpy import prod

def fun1(a_list):
    return prod(a_list)

if __name__ == "__main__":

    #Create FuzzyAggregator object and toggle verbose mode
    A = FuzzyAggregator(verbose=True)
    
    #Define some fuzzy sets for variables and set their name with "term"
    FS1 = FuzzySet(points=[[25,0], [100, 1]],   term="quality")
    FS2 = FuzzySet(points=[[30,1], [70, 0]],    term="price")

    #Add fuzzy sets objects to FuzzyAggregator
    A.add_variables(FS1,FS2)
    
    #Set numerical name of variables
    A.set_variable("quality", 55)
    A.set_variable("price", 42)
    
    #Perform aggregation. Available methods: product, min, max, arit_mean. Accepts pointer to an aggregation function.
    result = A.aggregate(["quality", "price"], aggregation_fun=fun1)

    print("Result:", result)
