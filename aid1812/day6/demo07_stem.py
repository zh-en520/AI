"""
demo07_stem.py 词性还原
"""
import nltk.stem as ns

words = ['table', 'probably', 'wolves', 
	'playing', 'is', 'dog', 'the', 
	'beaches', 'grounded', 'dreamt',
	'envision']

lemmatizer = ns.WordNetLemmatizer()

for word in words:
	n_lemma = lemmatizer.lemmatize(
			word, pos='n')
	v_lemma = lemmatizer.lemmatize(
			word, pos='v')
	print( '%8s %8s %8s' % (
		word, n_lemma, v_lemma) )


