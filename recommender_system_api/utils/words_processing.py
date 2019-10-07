from rake_nltk import Rake
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def words_processing(df, stopwords_path):
    """
    Combine all text description in to one columns
    :param stopwords_path: stopwords file path
    :param df: DataFrame: coupon or vendor dataframe
    :param type: String: 'coupon' or otherwise ('vendor')
    :return: DataFrame: with 2 columns ['coupon_name', 'bag_of_words'] for coupon
    or ['vendor_name', 'bag_of_words'] for vendor
    """

    df['bag_of_words'] = df.agg(' '.join, axis=1)

    bag_of_words_series = df.bag_of_words

    stopwords_en = get_stop_words(stopwords_path + 'stopwords-en.txt')
    stopwords_ja = get_stop_words(stopwords_path + 'stopwords-ja.txt')
    stopwords_zh = get_stop_words(stopwords_path + 'stopwords-zh.txt')
    stopwords = stopwords_en + stopwords_ja + stopwords_zh
    r = Rake(stopwords=stopwords)
    indices = bag_of_words_series.index

    for i in indices:
        bag_of_words = bag_of_words_series.at[i]
        r.extract_keywords_from_text(bag_of_words)
        keywords = r.get_ranked_phrases()
        keywords = ' '.join(keywords)
        bag_of_words_series.at[i] = keywords

    return bag_of_words_series


def get_stop_words(stopwords_path):
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = f.readlines()
        stop_list = list(set(m.strip() for m in stopwords))
        return stop_list


def count_vector(bag_of_words):
    count = CountVectorizer(lowercase=True, max_df=0.85, max_features=10000)
    count_matrix = count.fit_transform(bag_of_words)
    return count_matrix


def tfidf(bag_of_words):
    # stopwords = get_stop_words(stopwords_path)
    tfidf = TfidfVectorizer(lowercase=True, max_df=0.85, max_features=10000)
    tfidf_matrix = tfidf.fit_transform(bag_of_words)
    return tfidf_matrix


def cosine_similar(matrix):
    cosine_sim = cosine_similarity(matrix, matrix)
    return cosine_sim