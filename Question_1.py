def insert_in_middle(arr, element):
    mid = len(arr) // 2
    arr.insert(mid, element)
    return arr

def remove_element(arr, element):    
    if element in arr:                   
        arr.remove(element)              
        print(f"{element} removed successfully.")  
    else:                                
        print(f"{element} not found.")  
    return arr

def traverse(arr):
    print("Array elements:")             
    for i, val in enumerate(arr):       
        print(f"Index {i} -> {val}")     

arr = [10, 20, 30, 40, 50]

print("Original Array:", arr)

arr = insert_in_middle(arr, 99)
print("After Insert in Middle:", arr)

arr = remove_element(arr, 30)
print("After Removing 30:", arr)         

traverse(arr)

