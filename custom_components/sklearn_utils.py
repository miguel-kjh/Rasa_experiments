
import pandas as pd
import re

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from wordcloud import STOPWORDS
from nltk.stem import PorterStemmer


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD # LSA
from sklearn.preprocessing import StandardScaler, Normalizer
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer

class SklearnUtils:

    def __init__(self) -> None:
        self.stemmer = PorterStemmer()
        self.REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        self.BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        self.REMOVE_NUM = re.compile('[\d+]')
        self.STOPWORDS = set(stopwords.words('english'))

    def _dialogue2DataFrame(self, dialogues: list) -> pd.DataFrame:
        
        df = {
            'Text': [],
            'Action': []
        }
        
        text = ''
        act  = ''
        for events in dialogues:
            events = events.as_dialogue().as_dict()
            for event in events['events'][1:]:

                if 'text' in event.keys():
                    text = event['text']
                    continue
                elif 'name' in event.keys():
                    act  = event['name']
                else:
                    continue

                df['Text'].append(text)
                df['Action'].append(act)
        
        df = pd.DataFrame(df)
        df = df[df['Action'] != 'action_listen']
        return df


    def _DataFrame2Tfidf_matrix(self, df: pd.Series) -> tuple:
        vectorizer = TfidfVectorizer(sublinear_tf= True, min_df=10, norm='l2', ngram_range=(1, 2), stop_words='english')
        X_train_vc = vectorizer.fit_transform(df)

        return pd.DataFrame(X_train_vc.toarray(), columns=vectorizer.get_feature_names()), vectorizer

    def _clean_text(self, text: str):
        """
        text: a string
        return: modified initial string
        """

        text_origin = text
        # lowercase text
        text = text.lower()

        # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = self.REPLACE_BY_SPACE_RE.sub(' ', text)

        # Remove the XXXX values
        text = text.replace('x', '')

        # Remove white space
        text = self.REMOVE_NUM.sub('', text)

        #  delete symbols which are in BAD_SYMBOLS_RE from text
        text = self.BAD_SYMBOLS_RE.sub('', text)

        # delete stopwords from text
        text = ' '.join(word for word in text.split() if word not in STOPWORDS)

        # removes any words composed of less than 2 or more than 21 letters
        text = ' '.join(word for word in text.split() if (len(word) >= 2 and len(word) <= 21))

        # Stemming the words
        text = ' '.join([self.stemmer.stem(word) for word in text.split()])

        return text if text else text_origin

    def transforms(self, dialogues: list) -> tuple:
        df_data = self._dialogue2DataFrame(dialogues)
        df_data['Text_clean'] = df_data['Text'].apply(self._clean_text)
        tf_idf_data, self.tf_idf_encoder = self._DataFrame2Tfidf_matrix(df_data['Text_clean'])
        return tf_idf_data, df_data['Action']

    def get_encoder(self):
        return self.tf_idf_encoder
