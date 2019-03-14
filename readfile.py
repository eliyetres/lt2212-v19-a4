import re


def readfile(eng_file, fre_file):
    english_words = []
    french_words = []
    with open(eng_file, encoding="utf8") as engtext:
        with open(fre_file, encoding="utf8") as fretext:
            for eng_line, fre_line in zip(engtext, fretext):
                eng_line = re.sub(
                    r'([\-\"\<\>\(\)\:\[\]\%\/\=\?\!\;\n\*\.\,]|[0-9])', '', eng_line).lower()
                fre_line = re.sub(
                    r'([\-\"\<\>\(\)\:\[\]\%\/\=\?\!\;\n\*\.\,]|[0-9])', '', fre_line).lower()               
                eng_words = eng_line.split(" ")
                fre_words = fre_line.split(" ")
                eng_line = filter(None, eng_line)
                fre_line = filter(None, fre_line)
                e = len(eng_words)
                f = len(fre_words)
                shortest_sentence = min(e, f)
                eng_words = eng_words[:shortest_sentence] # Cut the longer sentence
                fre_words = fre_words[:shortest_sentence]
                english_words.append(eng_words)
                french_words.append(fre_words)
                print(eng_words)
                print(fre_words)
    return english_words, french_words


#readfile("UN-english.txt", "UN-french.txt")
