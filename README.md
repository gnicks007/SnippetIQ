# SnippetIQ - Synopsis

Data:

Took 260 of the most popular medium articles with at least 100 social media shares, and extracted the sentences therein
for storage in a database / dataframe. Each sentence could then be tagged has being highlighted or not. 
In this way, I could convert the problem into one of classification. 

This method yielded a dataframe with 20,795 sentences each stored in a single dataframe row, with 23 total columns.  
20,625 remained when sentences with empty strings were eliminated. The number of sentences that were tagged as having 
highlights, or others denoted as engaging, were approximately 455. 

The columns in the dataframe were as follows: post date, post id, post number, highlighted, 
relative location of the sentence, relative length of the sentence, post content, read time or length of the article, 
the ordinal position of the sentence, the sentence text, the number of post images, article subtitle, title,
top highlighted text, post word count, post tags, response count, number of post tags, url of post, 
number of social media shares, number of post sentences, relative location of the post highlight,
the relative length of the highlight.

•	Post_date	Post_id	Post_num	isHighlight	relLocationSentence
•	relLengthSentence	post_content	read-time	sentence_num	sentence_text
•	image_count	subtitle	title	tophighlight
•	post_word_count	post_tags	response_count	number_post_tags
•	url	recommend_count	post_number_sentence	relLocationHighlight
•	relLengthHighlight


Text data was cleaned by tokenizing, lowering all cases, removing stop words, 
eliminating digits, and finally, lemmatizing which simply removes various inflections of a word. 
Text data in the dataframe were cleaned according to this procedure before vectorization or other processing.

Feature Engineering: Positional Features
•	With positional features such as:
o	Sentence location (sentence number)
o	Relative sentence location
o	Relative length of sentence
o	Length of sentence
o	Length of the article
