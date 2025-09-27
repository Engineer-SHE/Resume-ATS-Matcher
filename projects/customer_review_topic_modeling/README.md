# 🔍 Project 5: Customer Review Topic Modelling

This mini-project demonstrates how to uncover themes hidden inside product and service reviews using unsupervised learning. We use 
a small sample of cafe and ecommerce reviews and apply topic modelling to detect patterns such as **shipping delays**, **product 
quality**, and **customer service**.

## 📦 Dataset

The `data/sample_reviews.csv` file contains 20 anonymised example reviews collected from fictitious Amazon, Shopify, and Yelp 
listings. Each row has:

- `review_id`: Unique identifier for the review
- `source`: Platform where the review was posted
- `rating`: Star rating between 1 and 5
- `review_text`: Raw text of the review

## ⚙️ How it works

1. The script loads the review dataset into a pandas DataFrame.
2. Text is tokenised with `CountVectorizer`, removing English stop words and keeping the most informative terms.
3. A `LatentDirichletAllocation` (LDA) model learns a user-defined number of topics.
4. The script prints the top keywords for each topic and assigns the dominant topic label to every review.

## 🚀 Quick start

```bash
python projects/customer_review_topic_modeling/topic_modeling.py 
```

The script outputs:

- A table summarising the keywords that define each discovered topic.
- A sample of labelled reviews showing the most likely topic for each comment.
- Aggregated review counts per topic and sentiment rating.

Feel free to swap the dataset with your own CSV as long as it contains a `review_text` column.
