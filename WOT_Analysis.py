import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

text_file = open('D:/Working_Directory/1-8_WOT.txt', 'rb')
text_initial = text_file.read().decode(encoding='utf-8')
text_initial = text_initial.lower()
text = ""
for x in text_initial:
    if x in (" ", "\n", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
             "t", "u", "v", "w", "x", "y", "z"):
        text += x

corpus = text.split("the wheel of time\n\n")
# print(corpus[0])
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 1), max_df=.6, min_df=.01)

X = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names_out()

dense = X.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)

data = df.transpose()
column_names = []
for x in range(8):
    column_name = "Book" + str(x + 1)
    column_names.append(column_name)
data.columns = column_names
# Find the top 30 words said by each President
# top_dict = {}
# for c in range(4):
#    top = data.iloc[:, c].sort_values(ascending=False).head(30)
#    top_dict[data.columns[c]] = list(zip(top.index, top.values))
# Print the top 15 words said by each President
# for chapter, top_words in top_dict.items():
#    print(chapter)
#    print(', '.join([word for word, count in top_words[0:14]]))
#    print('---')
# print(data.head())
# print(data['Chapter3'][10:20])

# change the value to black


def black_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(0,100%, 1%)"


# set the wordcloud background color to white
# set max_words to 1000
# set width and height to higher quality, 3000 x 2000
# wordcloud = WordCloud(background_color="white", width=3000, height=2000, max_words=500).generate_from_frequencies(data['Book1'])
# set the word color to black
# wordcloud.recolor(color_func = black_color_func)
# set the figsize
# plt.figure(figsize=[15,10])
# plot the wordcloud
# plt.imshow(wordcloud, interpolation="bilinear")
# remove plot axes
# plt.axis("off")
# save the image
# plt.savefig('WOT_Book1.png')

# english_words_file = open('D:/Working_Directory/English_Words/1-1000.txt', 'rb')
# english_words = english_words_file.read().decode(encoding='utf-8')

# words = text.split()
# unique = []
# for word in words:
    # if word not in english_words:
#    if word not in unique:
#        unique.append(word)

# word_count = [[], []]

# for x in unique:
#    if words.count(x) > 500:
#        word_count[0].append(words.count(x))
#        word_count[1].append(x)

# word_count[0].sort(reverse=True)

# plt.bar(word_count[1], word_count[0])
# plt.title('WOT Word Count')
# plt.xlabel('Words')
# plt.ylabel('Word Count')
# plt.show()
# ------------------------------------------


