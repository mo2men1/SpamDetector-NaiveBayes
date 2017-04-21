import operator
import math

STOP_WORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

Spam = []
Ham = []
vocab = {}
nSpam = 0;
nHam = 0;
nWords_Spam = 0
nWords_Ham = 0
with open('data/train', 'r') as f:
	for line in f.readlines():
		id, type, *word_arr = line.split(" ")
		words = {}
		for i, word in enumerate(word_arr):
			if i%2 == 0:
				if word.lower() in STOP_WORDS:
					continue
				n = int(word_arr[i+1].split("\n")[0])	#split to remove the \n in the last number
				words[word] = n
				if not word in vocab:
					vocab[word] = {"total": 0, "spam": 0, "ham": 0}
				vocab[word]["total"] +=  n
				if type == "ham":
					vocab[word]["ham"] += n
					nWords_Ham += n
				else:
					vocab[word]["spam"] += n
					nWords_Spam += n
		if type == "ham":
			nHam += 1
			Ham.append({"id": id, "type": type, "words": words})
		else:
			nSpam += 1
			Spam.append({"id": id, "type": type, "words": words})


pHam = nHam / (len(Spam) + len(Ham))
pSpam = nSpam / (len(Spam) + len(Ham))

P_W_given_C = {}
for word, value in vocab.items():
	P_W_given_SPAM = (value["spam"] + 1) /(nWords_Spam + len(vocab))
	P_W_given_HAM = (value["ham"] + 1) /(nWords_Ham + len(vocab))
	P_W_given_C[word] = {}
	P_W_given_C[word]["spam"] = P_W_given_SPAM
	P_W_given_C[word]["ham"] = P_W_given_HAM

sorted_SPAM = sorted(P_W_given_C.items(), key=lambda x_y: x_y[1]['spam'], reverse=True)
sorted_HAM = sorted(P_W_given_C.items(), key=lambda x_y: x_y[1]['ham'], reverse=True)
top_5_spam = list(map(operator.itemgetter(0), sorted_SPAM[0:5]))
top_5_ham = list(map(operator.itemgetter(0), sorted_HAM[0:5]))

print("The top 5 spam words are ", ", ".join(top_5_spam))
print("The top 5 ham words are ", ", ".join(top_5_ham))

accuracy = 0
count = 0
with open('data/test', 'r') as f:
	for line in f.readlines():
		count += 1
		likelihoods_product_spam = 1
		likelihoods_product_ham = 1
		id, type, *word_arr = line.split(" ")
		for i, word in enumerate(word_arr):
			if word not in vocab:
				continue

			if i%2 == 0:
				# likelihoods_product_ham *= P_W_given_C[word]["ham"]
				# likelihoods_product_spam *= P_W_given_C[word]["spam"]
				likelihoods_product_ham += math.log(P_W_given_C[word]["ham"])
				likelihoods_product_spam += math.log(P_W_given_C[word]["spam"])

		# P_ham = pHam * likelihoods_product_ham
		# P_spam = pSpam * likelihoods_product_spam
		P_ham = math.log(pHam) + likelihoods_product_ham
		P_spam = math.log(pSpam) + likelihoods_product_spam
		res = "ham" if P_ham > P_spam else "spam"
		# print("result: %s, ham: %d, spam: %d" % (res, P_ham, P_spam))
		if res == type:
			accuracy += 1
accuracy = accuracy / count * 100
print("Accuracy is: %.2f%%" % accuracy)
