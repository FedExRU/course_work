
import os
from symspellpy.symspellpy import SymSpell, Verbosity  # import the module
#import time
from pprint import pprint
from langdetect import detect

lang = detect("html")

print(lang)

def main():
	#start_time = time.clock()
	initial_capacity = 425361
	max_edit_distance_dictionary = 2
	prefix_length = 7
	sym_spell = SymSpell(initial_capacity, max_edit_distance_dictionary,prefix_length)
	dictionary_path = os.path.join(os.path.dirname(__file__), "data/ru.txt")
	
	term_index = 0  # column of the term in the dictionary text file
	count_index = 1  # column of the term frequency in the dictionary text file
	if not sym_spell.load_dictionary(dictionary_path, term_index, count_index):
		print("Dictionary file not found")
		return

	max_edit_distance_lookup = 2
	suggestion_verbosity = Verbosity.CLOSEST

	#input_term = ("whereis th elove hehad dated forImuch of thepast who couqdn'tread in sixtgrade and ins pired him")
	#input_term2 = ("HTML JS JavaScript Linux, avito CSS интерактивный")
	#input_term2 = ("Опыт в процессе планирования, оценки сроков и стоимости 1C разработки")
	
	input_term1 = ("Опыт")
	splittext = input_term1.split(" ")
	correctWords = []
	max_edit_distance_lookup = 2
	for i in splittext:
		lang = detect(i)
		print(lang)
		print(i)
		print("__")
		if(lang == 'ru'):
			word = sym_spell.lookup_compound(i, max_edit_distance_lookup)
			for y in word:
				correctWords.append(format(y.term))

	suggestions1 = ' '.join(correctWords)
	pprint(suggestions1)


	# suggestions2 = sym_spell.lookup_compound(input_term2, max_edit_distance_lookup)
	

	# for suggestion in suggestions1:
	# 	print(format(suggestion.term))

	# for suggestion in suggestions2:
	# 	print(format(suggestion.term))

	#print(time.clock() - start_time)

if __name__ == "__main__":
	main()
