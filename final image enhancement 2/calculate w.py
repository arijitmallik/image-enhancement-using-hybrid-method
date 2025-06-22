def calw(iq):
    # Define the iq values as a list
    iq_values = [
         3.810  
,5.361 , 
6.569 , 
7.897,  
8.754,  
2.719,  
1.892,  
1.537  ,
1.352  ,
1.202  

    ]
    
    # Sum of all iq values
    total_sum = sum(iq_values)
    
    # Calculate weight: iq divided by the sum of all iq values (normalization)
    result = iq / total_sum
    
    return result

# Test the function with iq = 2.077434
result = calw(1.202,)
print(f"Result for : {result:.6f}")