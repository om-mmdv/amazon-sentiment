\# Amazon Review Sentiment Analysis



\*\*Live demo:\*\* https://amazon-sentiment-bput.streamlit.app/

Binary sentiment classification of Amazon Fine Food Reviews with TF-IDF and

linear models, a leak-free evaluation, an analysis of errors and an

interactive Streamlit dashboard.



!\[Dashboard](assets/dashboard.png)



Dataset: \[Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

(Kaggle, Stanford SNAP), 568,454 reviews. The CSV is not included.



\## Setup of the task



\- Label from stars: 4-5 = positive, 1-2 = negative. 3-star reviews are dropped.

\- Negative is the target class (15.6% of reviews), so we report precision,

&#x20; recall, F1 and PR-AUC for negatives instead of accuracy

&#x20; ("always positive" would give 84% accuracy).

\- \*\*30.8% of review texts were duplicates\*\* (the same review under several

&#x20; product ids). They were removed.

\- The split is made \*\*by user\*\*, so all reviews of one author are either in

&#x20; train or in test. After cleaning: 364,910 reviews.



\## Results (test set, negative class)



| Model | Threshold | Precision | Recall | F1 | PR-AUC |

|---|---|---|---|---|---|

| Always positive | - | 0.000 | 0.000 | 0.000 | 0.156 |

| Logistic regression, threshold 0.5 | 0.50 | 0.759 | 0.934 | 0.837 | 0.930 |

| Logistic regression | 0.36 | 0.854 | 0.880 | 0.867 | 0.936 |

| \*\*Linear SVM (calibrated)\*\* | 0.38 | \*\*0.865\*\* | \*\*0.890\*\* | \*\*0.877\*\* | \*\*0.943\*\* |

| Complement Naive Bayes | 0.62 | 0.785 | 0.859 | 0.820 | 0.882 |



Thresholds were chosen on a validation part of the training set (best F1),

never on the test set. The gap between SVM and logistic regression is small

and should not be over-interpreted.



\## What the model learned



Words pushing toward negative: \*terrible, disappointed, awful, worst, stale,

bland, not good, not worth\*. Toward positive: \*great, delicious, excellent,

perfect, yummy\*. Bigrams capture negation (\*not good\*).



!\[Negative words](assets/top\_negative.png)



!\[Positive words](assets/top\_positive.png)



\## Errors: noisy labels



Many "mistakes" are label noise rather than model failures. Reviews with text

like "The candy was so delicious" carry 1-2 stars, and reviews like "The food

was disgusting" carry 4-5 stars. The model is very confident on these

(scores of exactly 0.00 or 1.00), so the true quality is probably higher than

the reported metrics. The label is the star rating, not the true sentiment

of the text.



\## Dashboard



```bash

streamlit run app.py

```



Type a review or pick an example. The dashboard shows the verdict, the

probability of negative sentiment and the words that drove it (TF-IDF value

times model weight). It uses logistic regression (`models/sentiment.joblib`,

3 MB) because its weights are interpretable.



\## Installation



```bash

git clone https://github.com/om-mmdv/amazon-sentiment.git

cd amazon-sentiment

python -m venv venv

venv\\Scripts\\activate

pip install -r requirements.txt

```



Put `Reviews.csv` into `data/`, then:



```bash

python -m src.explore

python -m src.baseline

python -m src.analyze

python -m src.compare

python -m src.export\_model

streamlit run app.py

```



\## Limitations



\- Bag-of-words ignores word order beyond bigrams: sarcasm and mixed reviews

&#x20; ("tasty, but arrived broken") can fool the model.

\- Labels come from star ratings and are noisy.

\- Reviews are about food only; the model will not transfer well to other

&#x20; product categories.

\- English only.



\## Possible improvements



\- Fine-tune a transformer (e.g. DistilBERT) and compare with TF-IDF

\- Predict the 1-5 star rating instead of a binary label

\- Clean noisy labels by flagging reviews where text and stars disagree

