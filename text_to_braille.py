# This is a simple text to braille converter just for taking outputs from the LLM and convert it to braille

'''Braille - there are 2 types of braille - grade 1 and grade 2.
Grade 1 braille translates each charachter into a braille charachter while grade 2 has 
shorthands meaning words that are reaoaetedly used have shortcuts or they are represented by 
a single or multiple braille charachters instead of one charachter for each letter.
----------------------------------------------------------------------
Here the conversion is limited to Grade 1 and if time permits we can upgrade it to grade 2 also
----------------------------------------------------------------------

How grade 1 braille works
-------------------------------
For each english alphabet there is a unique braille charachter. Braille is represented by 6 dots.
For numbers, the charachters coincide with first 10 alphabets - so to differentiaate between them,
a number sign charachter is put before the numbers.
Also for capital letters , a capital sign braille charachter is inserted before the letter
-----------------------------------------
'''

import numpy as np


#This is a dict that maps each english alphabet, numeriacals and some signs to the braille.
#Of course the charachters are not completed. It needs much more addition.

text_to_braille_mapping = {
    #Letters
    ' ': [chr(0x2800)],    #These are  the Unicode for the braille charachters
    'a': [chr(0x2801)],     #They are represented as lists because - some cgaracgeters need 2 braille charachters to represent them in braille.
    'b': [chr(0x2803)],
    'c': [chr(0x2809)],
    'd': [chr(0x2819)],
    'e': [chr(0x2811)],
    'f': [chr(0x280B)],
    'g': [chr(0x281B)],
    'h': [chr(0x2813)],
    'i': [chr(0x280A)],
    'j': [chr(0x281A)],
    'k': [chr(0x2805)],
    'l': [chr(0x2807)],
    'm': [chr(0x280D)],
    'n': [chr(0x281D)],
    'o': [chr(0x2815)],
    'p': [chr(0x280F)],
    'q': [chr(0x281F)],
    'r': [chr(0x2817)],
    's': [chr(0x280E)],
    't': [chr(0x281E)],
    'u': [chr(0x2825)],
    'v': [chr(0x2827)],
    'w': [chr(0x283A)],
    'x': [chr(0x282D)],
    'y': [chr(0x283D)],
    'z': [chr(0x2835)],

    #Capital Letters
    'A': [chr(0x2820),chr(0x2801)],
    'B': [chr(0x2820),chr(0x2803)],
    'C': [chr(0x2820),chr(0x2809)],
    'D': [chr(0x2820),chr(0x2819)],
    'E': [chr(0x2820),chr(0x2811)],
    'F': [chr(0x2820),chr(0x280B)],
    'G': [chr(0x2820),chr(0x281B)],
    'H': [chr(0x2820),chr(0x2813)],
    'I': [chr(0x2820),chr(0x280A)],
    'J': [chr(0x2820),chr(0x281A)],
    'K': [chr(0x2820),chr(0x2805)],
    'L': [chr(0x2820),chr(0x2807)],
    'M': [chr(0x2820),chr(0x280D)],
    'N': [chr(0x2820),chr(0x281D)],
    'O': [chr(0x2820),chr(0x2815)],
    'P': [chr(0x2820),chr(0x280F)],
    'Q': [chr(0x2820),chr(0x281F)],
    'R': [chr(0x2820),chr(0x2817)],
    'S': [chr(0x2820),chr(0x280E)],
    'T': [chr(0x2820),chr(0x281E)],
    'U': [chr(0x2820),chr(0x2825)],
    'V': [chr(0x2820),chr(0x2827)],
    'W': [chr(0x2820),chr(0x283A)],
    'X': [chr(0x2820),chr(0x282D)],
    'Y': [chr(0x2820),chr(0x283D)],
    'Z': [chr(0x2820),chr(0x2835)],

    #Numbers
    '0': [chr(0x281A)],
    '1': [chr(0x2801)],
    '2': [chr(0x2803)],
    '3': [chr(0x2809)],
    '4': [chr(0x2819)],
    '5': [chr(0x2811)],
    '6': [chr(0x280B)],
    '7': [chr(0x281B)],
    '8': [chr(0x2813)],
    '9': [chr(0x280A)],

    #Extra charachters - Not all are included
    ',': [chr(0x2802)],
    '.': [chr(0x2832)],
    '?': [chr(0x2826)],
    '!': [chr(0x2816)],
    "'": [chr(0x2804)],
    '-': [chr(0x2824)],
    '(': [chr(0x2836)],
    ')': [chr(0x2836)],
    '"': [chr(0x2834)],
    ':': [chr(0x2812)],
    ';': [chr(0x2806)],
    '/': [chr(0x2838)],
    ',': [chr(0x2802)],
    ',': [chr(0x2802)],
    ',': [chr(0x2802)],
    '\ue000': [chr(0x283C)],   #number_sign for braille
    'capital_sign': [chr(0x2820)],  #capital_sign for braille

    '\n': ['\n']  #This is to keep the formatting of the input text, so the paragraphs stay intact

}

#'\ue000' is called a private use area(PUA). this refers to codepoints in unicode 
#that are not assigned to any charachter - they are reserved for private use
#The reason for its use will be understood by reading the full code.


#Also the translation will be according to bharath braille system.


input = '''Machine Learning \n
123 568 967 -- '''


#Converting the input to a list of letters
#list comprehension is used because it is much faster
input_letters = [letter for letter in input]

#Converting the list to a numpy array - to easily apply transformations for each element
input_array = np.array(input_letters)

#So we need to place a number sign before every number, that is it should be placed
#only before the first digit - not before every number

#the bool array is a boolean array showing true for numbers and false for non numbers
bool_array = np.isin(input_array, ['0','1','2','3','4','5','6','7','8','9'])


index_array= np.where(bool_array) #it returns the indexes where True is present.
#That is it literally returns the indexes of numbers present in the input.


# The below is a code to extract the index positions of only the first digit from the list of indexes of all numbers
indexes_needed = []
for i in range(len(index_array[0])):   #index array[0] is used as index array is 2d.
  if i == 0:
    indexes_needed.append(index_array[0][i])
    continue
  if index_array[0][i-1] != index_array[0][i] - 1:
    indexes_needed.append(index_array[0][i])

final_array = np.insert(input_array,indexes_needed,chr(0xE000))
#this will insert the PUA characheter befor the first digits- so that
#it will be replaced by number sign braille charachter in the coming codes


output  = np.fromiter((text_to_braille_mapping[x] for x in final_array), dtype='object')
#fromiter is simply used to create an array from an iterable like list - it is much more memory efficient

#The output now will be an array of lists. - as each letter was replaced by a list of braille charachters



result = np.concatenate(output) #This converts it into a single array

result = ''.join(result) #this joins it to a single string

print(result)

#we can check if the output is right by using an online braille converter
#also this code doesnt count for the error thaat will be caused due to a charachter that is not present in the mapping dict.