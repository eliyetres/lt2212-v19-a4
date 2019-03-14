import re

def readfile(filename):
    with open(filename, encoding="utf8") as textfile:
        for line in textfile:            
            line = re.sub(r'([\-\"\<\>\(\)\:\[\]\%\/\=])', '', line).lower()
            print(line)
            

#readfile("UN-french.txt")
readfile("UN-english.txt")