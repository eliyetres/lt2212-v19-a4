import re


def readfile(eng_file, fre_file):
    english_words = []
    french_words = []
    with open(eng_file, encoding="utf8") as engtext:
        with open(fre_file, encoding="utf8") as fretext:
            for eng_line, fre_line in zip(engtext, fretext):
                eng_line = re.sub(
                    r'([\"\<\>\(\)\:\[\]\%\/\=\?\!\;\n\*\.\,]|[0-9])', '', eng_line).lower()
                fre_line = re.sub(
                    r'([\"\<\>\(\)\:\[\]\%\/\=\?\!\;\n\*\.\,]|[0-9])', '', fre_line).lower()
                # Split string into list of words
                eng_words = eng_line.split(" ")
                fre_words = fre_line.split(" ")
                eng_words = [x for x in eng_words if x]  # Remove empty strings
                fre_words = [x for x in fre_words if x]
                e = len(eng_words)
                f = len(fre_words)
                shortest_sentence = min(e, f)
                # Cut the longer sentence
                eng_words = eng_words[:shortest_sentence]
                fre_words = fre_words[:shortest_sentence]
                english_words.append(eng_words)
                french_words.append(fre_words)
                
    return english_words, french_words
