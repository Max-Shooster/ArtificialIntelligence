"""
Name: Max Shooster
File: firstOrderLogic.py

Description: Implement a theorem prover for a subset of first order logic (FOL).
Given a single file encoding a knowledge base (KB) in conjunctive normal form
(CNF), the program should determine if the KB is satisfiable and print either,
"yes" (it is satisfiable; it cannot find any new clauses) or "no" (it is not
satisfiable; it finds the empty clause). 

Input: The program should take 1 argument, KB.cnf. 
e.g. python firstOrderLogic.py KB.cnf
"""
import os
import sys
import math

# clause
class Clause:
    def __init__(self, predicates):
        self.predicates = frozenset(predicates)
    def __hash__(self):
        return self.predicates.__hash__()
    def __eq__(self, x):
        return self.predicates == x.predicates
# variable
class Variable:
    def __init__(self, name):
        self.name = name
# function
class Function:
    def __init__(self, name, args):
        self.name = name
        self.args = args
# predicate    
class Predicate:
    def __init__(self, name, args, isNegated=False):
        self.name = name
        self.args = args
        self.isNegated = isNegated
    def __key(self):
        return (self.name, tuple(self.args), self.isNegated)
    def __hash__(self):
        return hash(self.__key())
    def __eq__(self, x):
        if self.name != x.name:
            return False
        if tuple(self.args) != tuple(x.args):
            return False
        if self.isNegated != x.isNegated:
            return False
        return True

# Parse Cnf input and return a list of clauses
def parseCnfInput(cnfFilePath):
    # open Cnf file to read
    with open(cnfFilePath, "r") as f:
        cnfLines = f.readlines()
    clausesList = [] # create empty list to store clauses
    # retrieve clauses from file and store in clauseLines
    idx = cnfLines.index("Clauses:\n")
    clauseLines = cnfLines[idx+1:]
    # iterate over the lines
    for line in clauseLines:
        predicatesList = [] # create empty list to store predicates
        for predicate in line.strip().split():
            isNegated = predicate.startswith("!") # check if negated
            predicate = predicate.strip("!")
            if "(" in predicate:
                # FOL
                name, args = predicate.split("(", 1)
                args = args[:-1]  # strip end parenthesis
                argsTmp = []
                for arg in args.split(","):
                    argsTmp.append(parseCnfInputHelper(arg.strip()))
                args = argsTmp
            else:
                # PROP
                name = predicate
                args = []
            predicatesList.append(Predicate(name, args, isNegated))
        clausesList.append(Clause(predicatesList))
    return clausesList # return list of clauses

# helper function to parse arg
def parseCnfInputHelper(arg):
    #function
    if "(" in arg:
        splitStr = arg[:-1].split("(", 1)
        funcStr = splitStr[0]
        argsStr = splitStr[1]
        args = []
        for argStr in argsStr.split(","):
            args.append(parseCnfInputHelper(argStr.strip()))
        return Function(funcStr, args)
    #variable
    elif arg.startswith(("x", "t", "l")):
        return Variable(arg)
    #constant
    else:
        return arg

# checks for satisfiability
def isSat(clauses):
    # initialize two sets to store clauses
    uncheckedClauses = set()
    checkedClauses = set() 
    while True:
        flag = False
        clausePairs = []
        for i in range(len(clauses)):
            for j in range(i+1, len(clauses)):
                clausePairs.append((clauses[i], clauses[j])) # creates all possible combinations
        for (clauseA, clauseB) in clausePairs:
            if (clauseA, clauseB) in checkedClauses: # skip if already seen this pair
                continue
            checkedClauses.add((clauseA, clauseB))
            resultClause, contradiction = isSatHelper(clauseA, clauseB) # attempt to resolve
            if contradiction:
                return "no"
            if resultClause:
                if resultClause not in uncheckedClauses:
                    flag = True
                    uncheckedClauses.add(resultClause)

        if not flag: # if satisfiable
            return "yes"
        clauses += list(uncheckedClauses)

# helper function for resolution to resolve two clauses
def isSatHelper(clauseA, clauseB):
    for predA in clauseA.predicates:
        for predB in clauseB.predicates:
            if isNegation(predA, predB): # if they are negations of each other
                subst = unify(predA.args, predB.args) # attempt to unify
                if subst is not None:
                    subA = doSub(clauseA.predicates-{predA}, subst)
                    subB = doSub(clauseB.predicates-{predB}, subst)
                    substPreds = subA+subB 
                    substPredsSet = set(substPreds)
                    if not substPredsSet: # then there is a contradiction
                        return Clause([]), True # return the empty clause
                    return Clause(substPredsSet), False
    return None, False

# helper function for isSatHelper to check if there is a negation
def isNegation(predA, predB):
    return predA.name == predB.name and predA.isNegated != predB.isNegated

# unification 
def unify(a, b, subst=None):
    if subst is None:
        subst = {}
    if a == b:
        return subst
    elif isinstance(a, Variable): # if a is a variable
        return unifyHelper(a, b, subst)
    elif isinstance(b, Variable): # if b is a variable
        return unifyHelper(b, a, subst)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b): # if a and b have same length list
        idx = 0
        while idx < len(a):
            subst = unify(a[idx], b[idx], subst)
            if subst is None:
                return None
            idx += 1
        return subst
    else:
        return None

# helper function for unification
def unifyHelper(var, x, subst):
    subst[var.name]=x
    return subst

# make a substitution
def doSub(predicates, subst):
    updatedPreds = []
    for pred in predicates:
        substArgs = []
        for arg in pred.args:
            substArgsResult = doSubHelper(arg, subst)
            substArgs.append(substArgsResult)
        updatedPreds.append(Predicate(pred.name, substArgs, pred.isNegated)) # create new predicate with the subst args
    return updatedPreds

# helper function to make a substitution
def doSubHelper(arg, subst):
    if isinstance(arg, Variable):
        if arg.name in subst:
            return subst[arg.name]
    elif isinstance(arg, Function):
        substArgs = []
        for a in arg.args:
            substArgsResult = doSubHelper(a, subst)
            substArgs.append(substArgsResult)
        return Function(arg.name, substArgs)
    return arg

def main(args):
    cnfFilePath = args[0] # file path
    clauses = parseCnfInput(cnfFilePath)
    # print(clauses)
    solution = isSat(clauses) # checks for satisfiability
    print(solution) # yes or no

if __name__ == "__main__":
    main(sys.argv[1:]) # cnf file path
        
# def main():
#     cnfFolderPath = sys.argv[1]
#     cnfFiles = [f for f in os.listdir(cnfFolderPath) if f.endswith(".cnf")]
#     cnfFiles.sort()

#     for f in cnfFiles:
#         cnfFilePath = os.path.join(cnfFolderPath, f)
#         clauses = parseCnfInput(cnfFilePath)
#         solution = isSat(clauses)
#         print(f"File: {f}, Sat: {'Yes' if solution == 'yes' else 'No'}")

# if __name__ == "__main__":
#     main()