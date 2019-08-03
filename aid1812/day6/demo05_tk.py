"""
demo05_tk.py 文本分词
"""
import nltk.tokenize as tk

doc = "Are you curious about tokenization? " \
	"Let's see how it works! " \
	"We need to analyze a couple of sentences " \
	"with punctuations to see it in action."
print(doc)
# 拆分文档得到句子列表
print('-' * 50)
sents = tk.sent_tokenize(doc)
for i, s in enumerate(sents):
	print(i+1, s)

# 拆分句子得到单词列表
print('-' * 50)
words = tk.word_tokenize(doc)
for i, s in enumerate(words):
	print(i+1, s)

print('-' * 50)
tokens = tk.WordPunctTokenizer()
words = tokens.tokenize(doc)
for i, s in enumerate(words):
	print(i+1, s)

