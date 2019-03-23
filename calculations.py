#Calculations.py

def calculate_crossentropy(log_probas):

    #Calculates the cross entropy of a probability distribution                                                                                                                     
    #The probability distribution is all (log) probabilities for all ngrams                                                                                                         
    #It is then divided with 1/N                                                                                                                                                    

    values = []

    #Sum the logs for all samples, by Wikipedia:                                                                                                                       
    for sample in log_probas:
        for log in sample:
            values.append(log)

    n = len(values)
    logsum = sum(values)
    crossentropy = -(1/n) * logsum

    return crossentropy
