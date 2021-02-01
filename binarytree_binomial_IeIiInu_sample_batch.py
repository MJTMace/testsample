# Script to check all possible tree outcomes are being output by batch run in the expected fractions (as barplot) 

import argparse
import numpy as np
from scipy.stats import binom
from binarytree import Node # see https://pypi.org/project/binarytree/ and https://github.com/joowani/binarytree/blob/master/binarytree/__init__.py

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D # for legend labels
import sys

def nodify(k):
    """
    Format charging events so can be printed nicely with binarytree's Node function
    """
    # Node method cannot handle non-numbers so I need to hack a way to display the plasma electron, plasma ion, and photoelectron event numbers at each level
    # k = [k_e, k_i, k_nu]
    return float(str(k[0]) + "." + str(k[1]) + str(k[2]) ) # note that if k_e=1, k_i=2, k_nu = 0 then it will read as "1.2" rather than "1.20"

def binomial_prob_multiI(k_arr, R): 
    """
    Works out how to resolve k>1 charging events in time across step
    Inputs: 
        k_arr = Number of charging events generated from Poisson distributions for multiple current types for the full timestep
        R = Random number drawn from uniform probability distribution U[0,1] to compare to the binomial probabilities
    """
            
     # Use the drawn uniform number to see how the charging events are split between the parent's 2 child sub-steps
     #Unpack the total number of charging events from the list passed into this function:
    N_e = k_arr[0]
    N_i = k_arr[1]
    N_nu = k_arr[2]

    #print(f"N_e = {N_e}, N_i = {N_i}, N_nu = {N_nu}")
    if N_e == 0 and N_i == 0 and N_nu == 0:
        print("WARNING: all N_e and N_i and N_nu = 0 !!")
        print("k_arr = ", k_arr)
        return k_arr
    elif N_e == 1 and N_i == 1 and N_nu == 1: # if there happens to be one electron and one ion charging event and one photoemission event, it should be equally probable to pick either of the 3
        #print("In special case [1,1,1]")
        k_ctr = [99999999999999, 99999999999999, 99999999999999] # Initialize variable: use redonkulus value rather than NaN, otherwise Node() complains!
        if R <= (1.0/3.0):
            k_ctr[0] = 1; k_ctr[1] = 0; k_ctr[2] = 0
        elif (1.0/3.0) < R <= (2.0/3.0):
            k_ctr[0] = 0; k_ctr[1] = 1; k_ctr[2] = 0
        else: # if R lies between 2/3 and 1
            k_ctr[0] = 0; k_ctr[1] = 0; k_ctr[2] = 1
    else:
        #print(f"In else clause, R = {R}")
        # Initialise 2-element array to store result for [I_e, I_i] number of charging events
        k_ctr=[99999999999999, 99999999999999, 99999999999999] # use an integer value that would get clearly flagged up if there's a bug in my code
        
        # Check where R falls in the weighted nested Binomial probability distribution
        # Note that because there is more than one type of current, I need to concurrently work out the respective probabilities for each type
        # Therefore I need to define separate counters k_e_ctr, k_i_ctr, k_nu_ctr
        
        ##First define plasma electron current probability variables
        start_pt_e = 0.0 # define a variable to store the start point of probability distribution range for I_e
        k_e_ctr = 0
        # Define the end point for the first partition (for X_e = 0) using prob mass function to calculate binomial probability for k_e=0, making sure to weight it appropriately
        end_pt_e = (N_e/(N_e+N_i+N_nu))*binom.pmf(k=k_e_ctr, n=N_e, p=0.5)
        
        #Rather than generating the full Binomial probability ranges, instead perform "Just in time" calculations, so that unnecessary calculations are avoided
        # First check whether it's an I_e type event:
        
        # Bug fix: if there happens to be one electron charging event and R falls inside of the k_e events bin then R must fall inside the k_e=1 event prob range
        if N_e == 1 and R < end_pt_e: 
            #print("In if N_e == 1")
            k_ctr[0]=1; k_ctr[1]=0; k_ctr[2]=0
            return(k_ctr)
            
        while ((R >= end_pt_e) and (k_e_ctr < N_e)):
            #print("In while ((R >= end_pt_e) and (k_e_ctr < N_e)) ")
            k_e_ctr+=1 #increment to the next prob range
            start_pt_e = end_pt_e # move start_pt to the next prob range bin's start (which is the end of the previous prob range's bin)
            end_pt_e += (N_e/(N_e+N_i+N_nu))*binom.pmf(k=k_e_ctr, n=N_e , p=0.5) # add on the appropriately weighted prob to update the end point of the probability bin

        # Store the number of I_e charging events as the 0th element in the list of k events:
        k_ctr[0] = k_e_ctr # note that if R doesn't lie within the I_e weighted range, then k_e_ctr is erroneously set to be N_e but it gets corrected below
        k_ctr[1] = 0 # If an electron event has been chosen then the ion event must be zero
        k_ctr[2] = 0 # If an electron event has been chosen then the photoelectron event must be zero
        #print(f"k_ctr = {k_ctr}")
        
        # Then check whether it's an I_i type event (only do this if R has not been found to lie inside I_e probability ranges)
        if R > end_pt_e: # only need to check the I_i prob ranges if R has proven to lie outside the I_e prob ranges
            #print("In if R > end_pt_e ")
            k_ctr[0] = 0 #if the code enters this if block then know that the plasma electron current has not been picked and so I need to overwrite with zero
            k_ctr[2] = 0 # also ensure that the photoelectron event is zero in the case of an ion event
                        
            # To save memory, only initialise variables if they are actually going to be used!
            # Define a variable to store the start point of probability distribution range for I_i current
            start_pt_i = N_e/(N_e+N_i+N_nu) # this coincides with the end point of the I_e current
            k_i_ctr = 0
            # Define the end point for the first partition (for X_i = 0) using prob mass function to calculate binomial probability for k_i=0
            end_pt_i = start_pt_i + (N_i/(N_e+N_i+N_nu))*binom.pmf(k=k_i_ctr, n=N_i, p=0.5)
            
            # Bug fix: if there happens to be one ion charging event and R does not fall inside any of the k_e events then R must fall inside the k_i event prob range
            if N_i == 1 and R < end_pt_i: 
                #print("In if N_i == 1")
                k_ctr[0]=0; k_ctr[1]=1; k_ctr[2]=0
                return(k_ctr)
            
            while ((R >= end_pt_i) and (k_i_ctr < N_i)):
                #print(f"In while ((R >= end_pt_i) and (k_i_ctr < N_i)) ")
                k_i_ctr+=1 #increment to the next prob range
                start_pt_i = end_pt_i # move start_pt to the next prob range bin's start (which is the end of the previous prob range's bin)
                end_pt_i += (N_i/(N_e+N_i+N_nu))*binom.pmf(k=k_i_ctr, n=N_i , p=0.5) # add on the appropriately weighted prob to update the end point of the probability bin

            # Store the number of I_i charging events as the 1st element in the list of k events:
            k_ctr[1] = k_i_ctr
        
            # Then check whether it's an I_nu type event (only do this if R has not been found to lie inside I_e or I_i probability ranges)
            if R > end_pt_i: # only need to check the I_i prob ranges if R has proven to lie outside the I_e prob ranges (note the indentation)
                k_ctr[0] = 0 #if the code enters this if block then know that the plasma electron current has not been picked and so I need to overwrite with zero
                k_ctr[1] = 0 # also ensure that the plasma ion current event is zero in the case of a photoelectron event
               
                # To save memory, only initialise variables if they are actually going to be used!
                # Define a variable to store the start point of probability distribution range for I_nu current
                start_pt_nu = (N_e+N_i)/(N_e+N_i+N_nu) # this coincides with the end point of the (I_e + I_i) currents
                k_nu_ctr = 0
                # Define the end point for the first partition (for X_i = 0) using prob mass function to calculate binomial probability for k_i=0
                end_pt_nu = start_pt_nu + (N_nu/(N_e+N_i+N_nu))*binom.pmf(k=k_nu_ctr, n=N_nu, p=0.5)
                 # Bug fix: if there happens to be one ion charging event and R does not fall inside any of the k_e events then R must fall inside the k_i event prob range
                if N_nu == 1 and R < end_pt_nu: 
                    k_ctr[0]=0; k_ctr[1]=0; k_ctr[2]=1
                    return(k_ctr)
                
                #print(f"start_pt_nu = {start_pt_nu}, end_pt_nu = {end_pt_nu}")
                while ((R >= end_pt_nu) and (k_nu_ctr < N_nu)):
                    #print(f"In while ((R >= end_pt_nu) and (k_nu_ctr < N_nu)) ")
                    k_nu_ctr+=1 #increment to the next prob range
                    start_pt_nu = end_pt_nu # move start_pt to the next prob range bin's start (which is the end of the previous prob range's bin)
                    end_pt_nu += (N_nu/(N_e+N_i+N_nu))*binom.pmf(k=k_nu_ctr, n=N_nu , p=0.5) # add on the appropriately weighted prob to update the end point of the probability bin
                    #print("R=", R, "Prob range for k_i = ", k_i_ctr, " is ", start_pt_i , "to ", end_pt_i )
                
                # Store the number of I_i charging events as the 1st element in the list of k events:
                k_ctr[2] = k_nu_ctr
                #print(f"k_ctr = {k_ctr}")
        
    return k_ctr


def walker(value_arr, level=0, tree=Node(0)): 
    #print("Inside walker()")
    """
    Binary tree walker. Recursively splits timestep in half and keeps track of which level it is on, until the base case of k = 0 or 1 is reached.
    Inputs:
        value = number of charging events at a particular tree level
        level = depth of tree 
        tree = object to be printed for visual check
    """
    
    if ( (value_arr[0]==1 and value_arr[1]==0 and value_arr[2]==0) or (value_arr[0] == 0 and value_arr[1] == 1 and value_arr[2]==0) or (value_arr[0] == 0 and value_arr[1] == 0 and value_arr[2]==1) or
        (value_arr[0]==0 and value_arr[1]==0 and value_arr[2]==0) or level == scriptArgs.level_lim): # base case (leaf nodes have no children)
        #print("Base case")
        tree.left = None 
        tree.right = None
        return [(value_arr, level)], tree # store the leaf node value (ie k=0 or k=1) and its corresponding level as tuple-element of list
    
    # Generate a random number from uniform prob distribution:
    R = np.random.uniform(low=0.0, high=1.0) 
    
    # Use binomial prob to determine how the number of charging events are split across 1st and 2nd half of step
    substep1_arr = binomial_prob_multiI(value_arr, R)  # value holds the list of [k_e, k_e] number of charging events
    #substep2_arr = np.array(value_arr) - np.array(substep1_arr) # Note that because value_arr is a numpy array as is substep1, I can do elementwise subtraction (ie treat the electron and ion counters separately)
    substep2_arr = [99999999999999, 99999999999999, 99999999999999] # intialise list (rather than mixing np array type)
    substep2_arr[0] = value_arr[0] - substep1_arr[0]
    substep2_arr[1] = value_arr[1] - substep1_arr[1]
    substep2_arr[2] = value_arr[2] - substep1_arr[2]

    #print(f"LEVEL = {level} R={R},  substep1 = {substep1_arr}")
    #print(f"substep2 = {substep2_arr}")
    
    # Define the left and child nodes:
    tree.left = Node(nodify(substep1_arr)) # use my hacky nodify function as Node cannot handle non-numbers (e.g. arrays)
    tree.right = Node(nodify(substep2_arr))
    
    a_arr, tree_l = walker(substep1_arr, level+1, tree.left) # recursive call to left child
    b_arr, tree_r = walker(substep2_arr, level+1, tree.right) # recursive call to right child
    
    #print(f"walker level{level} called with value {value_arr} splits into {substep1_arr} and {substep2_arr} as R = {R}")
    #print(f" left subtree a_arr = {a_arr}, right subtree b_arr = {b_arr}")
    
    # Reconfigure the binary tree to update:
    tree.left = tree_l
    tree.right = tree_r
    return a_arr + b_arr, tree #concatenate list of lists (a_arr+b_arr) and also return tree

def main(scriptArgs):
    lev = 0
    # In order to handle 3 different types of current, the numeric values following ctr_follows the pattern of starting at the left topmost leaf noe and going rightward... where the first value corresponds to I_e, the second value corresponds to I_i, the third value corresponds to I_nu
    # where I suffix underscore with letter to indicate what level : a = level 1, b = level 2 , c = level 3 , ... 
     
    # NOTE THAT ctr211_ refers to an initial number of charging events k_e=2, k_i=1, k_nu=1
    if scriptArgs.k_e_init == 2 and scriptArgs.k_i_init == 1 and scriptArgs.k_nu_init == 1: 
        if scriptArgs.level_lim == 1:
            #LEVEL 1
            ctr211_a000_a211 = 0; ctr211_a100_a111 = 0; ctr211_a200_a011 = 0; ctr211_a010_a201 = 0; ctr211_a001_a210 = 0; # Initialise counters 
        elif scriptArgs.level_lim == 2:
            # LEVEL 2
            ctr211_a000_b000_b211 = 0; ctr211_a000_b100_b111 = 0; ctr211_a000_b200_b011 = 0; ctr211_a000_b010_b201 = 0;  ctr211_a000_b001_b210 = 0 ;
            ctr211_a100_b100_b011 = 0; ctr211_a100_b010_b101 = 0; ctr211_a100_b001_b110 = 0;
            
            ctr211_b200_b000_b010_b001 = 0; ctr211_b200_b000_b001_b010 = 0;
            ctr211_b100_b100_b010_b001 =0; ctr211_b100_b100_b001_b010 = 0; ctr211_b000_b200_b010_b001 = 0; ctr211_b000_b200_b001_b010=0;
            
            ctr211_a010_b000_b201 = 0; ctr211_a010_b100_b101 = 0; ctr211_a010_b200_b001 = 0; ctr211_a010_b001_b200 = 0
            ctr211_a001_b000_b210 = 0; ctr211_a001_b100_b110 = 0; ctr211_a001_b200_b010 = 0; ctr211_a001_b010_b200 = 0
        else:
             print("only written test up to level 2!")
    else:
        print("only written tests for k_init = [2,1,1]! ")
        sys.exit(0)
    # Initialise  dictionary to store tree output results:
    dict_tree_rand = {}
   
    # Run random binary tree generator a number of times
    for i in np.arange(0, scriptArgs.N_runs):
        #print(i)
        leaf_nodes, tree = walker([scriptArgs.k_e_init, scriptArgs.k_i_init, scriptArgs.k_nu_init], level=lev, tree=Node(nodify([scriptArgs.k_e_init, scriptArgs.k_i_init, scriptArgs.k_nu_init])) ) # input into nodify() function as  array [k_e,k_i]
 
        # Keep track of binary tree outcomes by "manually" checking known predicted possible outcomes:
        # k_e = 2, k_i = 1, k_nu = 1
        if scriptArgs.k_e_init == 2 and scriptArgs.k_i_init == 1 and scriptArgs.k_nu_init == 1: # Initialise counters 
            # LEVEL 1
            if leaf_nodes == [ ([0, 0, 0], 1), ([2,1,1], 1)]: # The walker() function outputs [ ([k_e, k_i], level), ...]
                ctr211_a000_a211 += 1
                dict_tree_rand[str(tree)] = ctr211_a000_a211
                # Dictionaries are unsorted so I need to match the key2s for the randomly generated dict and the predicted freq dict
                pred_tree_key211_a000_a211 = str(tree) # hideously inefficient as I only need to do this once but oh well! 
            elif leaf_nodes == [([1,0,0], 1), ([1,1,1], 1)]:
                ctr211_a100_a111 += 1
                dict_tree_rand[str(tree)] = ctr211_a100_a111
                pred_tree_key211_a100_a111 = str(tree) # hideously inefficient but oh well! 
            elif leaf_nodes == [([2,0,0], 1), ([0,1,1], 1)]:
                ctr211_a200_a011 += 1
                dict_tree_rand[str(tree)] = ctr211_a200_a011
                pred_tree_key211_a200_a011 = str(tree) # hideously inefficient but oh well! 
            elif leaf_nodes == [([0,1,0], 1), ([2,0,1], 1)]:
                ctr211_a010_a201 += 1
                dict_tree_rand[str(tree)] = ctr211_a010_a201
                pred_tree_key211_a010_a201 = str(tree) # hideously inefficient but oh well!
            elif leaf_nodes == [([0,0,1], 1), ([2,1,0], 1)]:
                ctr211_a001_a210 += 1
                dict_tree_rand[str(tree)] = ctr211_a001_a210
                pred_tree_key211_a001_a210 = str(tree) # hideously inefficient but oh well!
            # LEVEL 2 
            elif leaf_nodes == [([0,0,0], 1), ([0,0,0],2), ([2,1,1],2)]:
                ctr211_a000_b000_b211 += 1
                dict_tree_rand[str(tree)] = ctr211_a000_b000_b211
                pred_tree_key211_a000_b000_b211 = str(tree)
            elif leaf_nodes == [([0,0,0], 1), ([1,0,0],2), ([1,1,1],2)]:
                ctr211_a000_b100_b111  += 1
                dict_tree_rand[str(tree)] = ctr211_a000_b100_b111 
                pred_tree_key211_a000_b100_b111  = str(tree)
            elif leaf_nodes == [([0,0,0], 1), ([2,0,0],2), ([0,1,1],2)]:
                ctr211_a000_b200_b011 += 1
                dict_tree_rand[str(tree)] = ctr211_a000_b200_b011
                pred_tree_key211_a000_b200_b011 = str(tree)
            elif leaf_nodes == [([0,0,0], 1), ([0,1,0],2), ([2,0,1],2)]:
                ctr211_a000_b010_b201 += 1
                dict_tree_rand[str(tree)] = ctr211_a000_b010_b201
                pred_tree_key211_a000_b010_b201 = str(tree)
            elif leaf_nodes == [([0,0,0],1), ([0,0,1],2), ([2,1,0],2)]:
                ctr211_a000_b001_b210 += 1
                dict_tree_rand[str(tree)] = ctr211_a000_b001_b210
                pred_tree_key211_a000_b001_b210 = str(tree)
            elif leaf_nodes == [([1,0,0], 1), ([1,0,0],2), ([0,1,1],2)]:
                ctr211_a100_b100_b011 += 1
                dict_tree_rand[str(tree)] = ctr211_a100_b100_b011
                pred_tree_key211_a100_b100_b011 = str(tree)
            elif leaf_nodes == [([1,0,0], 1), ([0,1,0],2), ([1,0,1],2)]:
                ctr211_a100_b010_b101 += 1
                dict_tree_rand[str(tree)] = ctr211_a100_b010_b101
                pred_tree_key211_a100_b010_b101 = str(tree)
            elif leaf_nodes == [([1,0,0], 1), ([0,0,1],2), ([1,1,0],2)]:
                ctr211_a100_b001_b110 += 1
                dict_tree_rand[str(tree)] = ctr211_a100_b001_b110 
                pred_tree_key211_a100_b001_b110  = str(tree)
            elif leaf_nodes == [([2,0,0],2), ([0,0,0],2), ([0,1,0],2), ([0,0,1],2)]:
                ctr211_b200_b000_b010_b001 += 1
                dict_tree_rand[str(tree)] = ctr211_b200_b000_b010_b001
                pred_tree_key211_b200_b000_b010_b001 = str(tree)
            elif leaf_nodes == [([2,0,0],2), ([0,0,0],2), ([0,0,1],2), ([0,1,0],2)]:
                ctr211_b200_b000_b001_b010 += 1
                dict_tree_rand[str(tree)] = ctr211_b200_b000_b001_b010
                pred_tree_key211_b200_b000_b001_b010 = str(tree)
            elif leaf_nodes == [([1,0,0],2), ([1,0,0],2), ([0,1,0],2), ([0,0,1],2)]:
                ctr211_b100_b100_b010_b001 += 1
                dict_tree_rand[str(tree)] = ctr211_b100_b100_b010_b001
                pred_tree_key211_b100_b100_b010_b001 = str(tree)
            elif leaf_nodes == [([1,0,0],2), ([1,0,0],2), ([0,0,1],2), ([0,1,0],2)]:
                ctr211_b100_b100_b001_b010 += 1
                dict_tree_rand[str(tree)] = ctr211_b100_b100_b001_b010
                pred_tree_key211_b100_b100_b001_b010 = str(tree)
            elif leaf_nodes == [([0,0,0],2), ([2,0,0],2), ([0,1,0],2), ([0,0,1],2)]:
                ctr211_b000_b200_b010_b001 += 1
                dict_tree_rand[str(tree)] = ctr211_b000_b200_b010_b001
                pred_tree_key211_b000_b200_b010_b001 = str(tree)
            elif leaf_nodes == [([0,0,0],2), ([2,0,0],2), ([0,0,1],2), ([0,1,0],2)]:
                ctr211_b000_b200_b001_b010 += 1
                dict_tree_rand[str(tree)] = ctr211_b000_b200_b001_b010
                pred_tree_key211_b000_b200_b001_b010 = str(tree)
            elif leaf_nodes == [([0,1,0],1), ([0,0,0],2), ([2,0,1],2)]:
                ctr211_a010_b000_b201  += 1
                dict_tree_rand[str(tree)] = ctr211_a010_b000_b201 
                pred_tree_key211_a010_b000_b201  = str(tree)
            elif leaf_nodes == [([0,1,0],1), ([1,0,0],2), ([1,0,1],2)]:
                ctr211_a010_b100_b101 += 1
                dict_tree_rand[str(tree)] = ctr211_a010_b100_b101 
                pred_tree_key211_a010_b100_b101  = str(tree)
            elif leaf_nodes == [([0,1,0],1), ([2,0,0],2), ([0,0,1],2)]:
                ctr211_a010_b200_b001  += 1
                dict_tree_rand[str(tree)] =  ctr211_a010_b200_b001 
                pred_tree_key211_a010_b200_b001 = str(tree)
            elif leaf_nodes == [([0,1,0],1), ([0,0,1],2), ([2,0,0],2)]:
                ctr211_a010_b001_b200  += 1
                dict_tree_rand[str(tree)] = ctr211_a010_b001_b200  
                pred_tree_key211_a010_b001_b200  = str(tree)
            elif leaf_nodes == [([0,0,1],1), ([0,0,0],2), ([2,1,0],2)]:
                ctr211_a001_b000_b210  += 1
                dict_tree_rand[str(tree)] =  ctr211_a001_b000_b210 
                pred_tree_key211_a001_b000_b210 = str(tree)
            elif leaf_nodes == [([0,0,1],1), ([1,0,0],2), ([1,1,0],2)]:
                ctr211_a001_b100_b110 += 1
                dict_tree_rand[str(tree)] =  ctr211_a001_b100_b110 
                pred_tree_key211_a001_b100_b110  = str(tree)
            elif leaf_nodes == [([0,0,1],1), ([2,0,0],2), ([0,1,0],2)]:
                ctr211_a001_b200_b010  += 1
                dict_tree_rand[str(tree)] = ctr211_a001_b200_b010 
                pred_tree_key211_a001_b200_b010 = str(tree)
            elif leaf_nodes == [([0,0,1],1), ([0,1,0],2), ([2,0,0],2)]:
                ctr211_a001_b010_b200  += 1
                dict_tree_rand[str(tree)] = ctr211_a001_b010_b200 
                pred_tree_key211_a001_b010_b200 = str(tree)
            else:
                print("leaf_nodes = ", leaf_nodes)
                print("There's a bug! Unexpected outcome.")
        else: 
            print("Choose a different k_init, as I've only written tests for k_init = [2,1,1]")
    #print("counters : ", ctr2_02, ctr2_11, ctr2_20)
    #print(dict_tree_rand)
  
    fig,ax = plt.subplots(figsize=(70,15))
    
    my_xticks = []
    freqs = []
    pred_freqs = [] # predicted frequency of particular tree outcomes 

    # Insert the predicted frequency of each  binary tree outcome (pen and paper calculations) 
    # The numeric value following pred_tree_key_ follows the pattern of starting at the left topmost node (excluding root) and going rightward, then down to the next level, ...etc
    dict_tree_pred = {}
    if scriptArgs.k_e_init == 2 and scriptArgs.k_i_init == 1 and scriptArgs.k_nu_init == 1: 
        # LEVEL 1
        if scriptArgs.level_lim == 1: # when considering output trees that go deeper than level 1, need to filter out certain trees which only hold for level 1 
            dict_tree_pred[pred_tree_key211_a000_a211] = 1/8 
            dict_tree_pred[pred_tree_key211_a100_a111] = 1/4  
            dict_tree_pred[pred_tree_key211_a200_a011] = 1/8
            dict_tree_pred[pred_tree_key211_a010_a201] = 1/4
            dict_tree_pred[pred_tree_key211_a001_a210] = 1/4
            # LEVEL 2
        elif scriptArgs.level_lim == 2:
            dict_tree_pred[pred_tree_key211_a000_b000_b211] = 1/64 
            dict_tree_pred[pred_tree_key211_a000_b100_b111] = 1/32 
            dict_tree_pred[pred_tree_key211_a000_b200_b011] = 1/64
            dict_tree_pred[pred_tree_key211_a000_b010_b201] = 1/32
            dict_tree_pred[pred_tree_key211_a000_b001_b210] = 1/32
   
            dict_tree_pred[pred_tree_key211_a100_b100_b011] = 1/12
            dict_tree_pred[pred_tree_key211_a100_b010_b101] = 1/12 
            dict_tree_pred[pred_tree_key211_a100_b001_b110] = 1/12
                
            dict_tree_pred[pred_tree_key211_b200_b000_b010_b001] = 1/64
            dict_tree_pred[pred_tree_key211_b200_b000_b001_b010] = 1/64
            dict_tree_pred[pred_tree_key211_b100_b100_b010_b001] = 1/32
            dict_tree_pred[pred_tree_key211_b100_b100_b001_b010] = 1/32
            dict_tree_pred[pred_tree_key211_b000_b200_b010_b001] = 1/64
            dict_tree_pred[pred_tree_key211_b000_b200_b001_b010] = 1/64
                
            dict_tree_pred[pred_tree_key211_a010_b000_b201] = 1/24
            dict_tree_pred[pred_tree_key211_a010_b100_b101] = 1/12
            dict_tree_pred[pred_tree_key211_a010_b200_b001] = 1/24
            dict_tree_pred[pred_tree_key211_a010_b001_b200] = 1/12

            dict_tree_pred[pred_tree_key211_a001_b000_b210] = 1/24
            dict_tree_pred[pred_tree_key211_a001_b100_b110] = 1/12
            dict_tree_pred[pred_tree_key211_a001_b200_b010] = 1/24
            dict_tree_pred[pred_tree_key211_a001_b010_b200] = 1/12
        else:
            print("only worked out tests for binary tree outcomes for k_e = 1, k_i = 2, k_nu = 1 up to level 2!")
    else: 
        print("only written tests for k = [2,1,1] ") 
    # Sort out lists of results:
    for key,value in dict_tree_rand.items():
        #print(key,value)
        my_xticks.append(key)
        freqs.append(value/scriptArgs.N_runs) # store frequency as fraction of total number of runs
        pred_freqs.append(dict_tree_pred[key])

    # Plot 
    x_placeholder = np.arange(start = 0 , stop = len(freqs)) # list of x-axis placeholder tick locations 
    ax.set_xticks(x_placeholder) # set the tick locations
    ax.bar(x_placeholder-0.15, freqs, color = "#657b83", width=0.3)
    ax.bar(x_placeholder+0.15, pred_freqs, color = "k", width=0.3)
    ax.set_xticklabels(np.array(my_xticks), fontsize=20)
    ax.tick_params(axis="both", which ='major', length = 10)
    ax.set_ylim([0,0.1])
    ax.set_xlabel("Binary tree output", fontsize=24)
    ax.set_ylabel("Normalised frequency", fontsize=24)
    #ax.set_yscale('log')
    # Create custom artsits for legend:
    #legend_elements = [Line2D([0], [0], color='#2d5986', lw=6, label=f'{scriptArgs.N_runs} random runs'),
    #                   Line2D([0], [0], color='#990000', lw=6, label='Prediction')]
    #ax.legend(handles=legend_elements, loc='upper right')

    fig.subplots_adjust(bottom=0.35)
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=16)   
    #fig.suptitle(r"Initial number of charging events, [$k_e, k_i, k_\nu$]  = " + f"[{scriptArgs.k_e_init}, {scriptArgs.k_i_init}, {scriptArgs.k_nu_init}], outcomes considered up to level {scriptArgs.level_lim}")
  
if __name__ == '__main__':
        #First create an argument parser argument:
        parser = argparse.ArgumentParser(description='Plot out frequency of outcomes for each possible binary tree given some initial charging events')
    
        #Filling an ArgumentParser with information about program arguments:
        parser.add_argument('-levLim', '--level_lim', help='Set max recursion lim', required=True, type=int) 
        parser.add_argument('-nrun', '--N_runs', help='Set number of runs', required=True, type=int) 
        parser.add_argument('-keinit', '--k_e_init', help='Set initial number of plasma electron current charging events', required=True, type=int) 
        parser.add_argument('-kiinit', '--k_i_init', help='Set initial number of plasma ion current charging events', required=True, type=int) 
        parser.add_argument('-knuinit', '--k_nu_init', help='Set initial number of photoemission current charging events', required=True, type=int) 
        # The object from parse_args() is a 'Namespace' object - An object whose member variables are named after your command-line arguments:
        scriptArgs = parser.parse_args()
        main(scriptArgs)
