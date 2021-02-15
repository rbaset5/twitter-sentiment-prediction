### Summary

The airline industry is a competitive market in the past 2 decades. Customer airlines still use traditioinal customer feedback forms which in turn can be time consuming and tedious. Twitter data can be used for competitive advantage around sentiment analysis of customers. 

This dataset comprises of tweets for 6 major US airlines where we performed a multi-class classification and sentiment analysis. 

This approach starts off with pre-processing techniques used to clean the tweets and then representing these tweets as vectors using a deep learning concept to do a phrase-level analysis using the Doc2Vec library. The analysis was carried out using 7 different classification strategies: Decision Tree, Random Forest, SVM, K-Nearest Neighbors, Logistic Regression, Gaussian Naïve Bayes and AdaBoost. The classifiers were trained using 80% of the data and tested using the remaining 20% data. The outcome of the test set is the tweet sentiment (positive/negative/neutral).

Based  on the results obtained, the accuracies were calculated to draw a comparison between each classification approach and the overall sentiment count was visualized combining all six airlines.

### Business problem 

Customer feedback is crucial to airline companies as it helps them in improving the quality of services and facilities provided to the customers. Sentiment analysis in airline industry is methodically done using traditional feedback methods that involve customer satistfaction questionares and forms. These procedures might seem quite simple on an overview but are very time consuiming and require a lot of manpower that comes with a cost in analyzing them. 

Moreover, the information collected from the questionnares may often be inaccurate and inconsistent.
This may be because not all customers take these feedbacks seriously and may fill in irrelevant details which result in noisy data for sentiment analysis. Whereas on the other hand, Twitter is a gold mine of data with over 1/60th of the world’s population using it which nearly amounts to 100 million people, more than half a billion tweets are tweeted daily and the number keeps growing with every passing day. With the rising demand and advancements of Big Data technologies in the past decade, it has become easier to collect tweets and apply data analysis techniques on them. 

Twitter is a much more reliable source of data as the users tweet their genuine feelings and feedbacks thus making it more suitable for investigation. 

For example, with the iPhone X market release, the company can perform a sentiment analysis on the tweets related to the product as a part of their market research to improvise their product. 

Once the airline tweets are collected, they undergo pre-processing to remove unecessary detials in them. Sentiment classifcation techniques are then applied to the cleaned tweets.

This gives data scientists and Airline companies a broader perspective about the feelings and opions of their customers. The main motive of this paper is to provide the airline industry a more comprehensive view about the sentiments of their customers an provide their needs in all good ways possible. In this paper, we go through several tweet pre-processing techniques followed by the appplication of seven different ML classificatoin algorithms that are used to deteremine the sentiment within the tweets. The classifiers are then compared agains each other for their accuracies. 
