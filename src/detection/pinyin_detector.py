import re
import jieba
import numpy as np

from typing import List
from tqdm   import tqdm

# from fuzzywuzzy import fuzz
# from fuzzywuzzy import process
from rapidfuzz import fuzz
from rapidfuzz import process

from pypinyin import pinyin, lazy_pinyin

from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.tokenize.sonority_sequencing import SyllableTokenizer


from src.utils import read_file

from src.detection.abs_detector import AbsDetector

tk = SyllableTokenizer()

class PinyinDetector(AbsDetector):
    def __init__(self, entity_path):
        self.entity_path = entity_path
        jieba.load_userdict(self.entity_path)

        self.entities, self.entities_syllable = self._load_entity(self.entity_path)
        
    def _load_entity(self, entity_path):
        entities = []
        entities = read_file(entity_path)
        entities = [e[0] for e in entities]
        entities = list(set(entities))
        entities_syllable = sorted([
            [self.word_to_syllable(entity)[0], entity] for entity in entities
        ])
        return entities, entities_syllable

    def remove_space(self, sent):
        sent = sent.split(' ')
        tmp  = []
        for s in sent:
            if s.replace(' ', '') == '':
                continue
            tmp.append(s)
        return ' '.join(tmp)

    def syllable(self, word, lang_type):
        if lang_type == 'english':
            syll = tk.tokenize(word)
        else:
            syll = lazy_pinyin(word)
        return syll

    def isEnglish(self, word):
        try:
            word.encode(encoding='utf-8').decode('ascii')
        except UnicodeDecodeError:
            return False
        else:
            return True

    def word_to_syllable(self, sentences):
        words = jieba.lcut(sentences, cut_all=False)
        word2syllable = []
        syllable2word = []
        sylls = []

        units = []
        for word in words:
            if word.replace(' ', '') == '':
                continue
            if self.isEnglish(word):
                units.append(word)
            else:
                units.extend(list(word))
        words = units
        for i in range(len(words)):
            word = words[i]
            lang_type = 'chinese' if not self.isEnglish(word) else 'english'
            syll = self.syllable(word, lang_type)
            word2syllable.append(len(syll))
            syllable2word.extend([i for _ in range(len(syll))])
            sylls.extend(syll)

        return self.remove_space(' '.join(sylls).lower()), word2syllable, syllable2word, words

    def sliding_window(self, sentence, sub_sentence, sub_word, threshold=0.6):
        sentence = sentence.split(' ')
        
        prob  = np.zeros([len(sentence)])
        count = np.zeros([len(sentence)])
        n_range = [1, 2, 3, 4, 5]
        n_range_prob = len(sub_sentence.split(" "))

        grams = []
        for n in n_range:
            n_grams  = ngrams(sentence, n)
            n_grams  = [[" ".join(gram), i, i + n] for i, gram in enumerate(list(n_grams))]
            grams.extend(n_grams)
        
        for gram in grams:
            sent, start, end = gram
            score = fuzz.ratio(sub_sentence, sent)
            score /= 100
            prob[start:end]  += score
            count[start:end] += 1
        
        prob /= count
        original_prob = prob.copy()
        scores = [[
                np.sum(gram) / n_range_prob, i, i + n_range_prob
            ] for i, gram in enumerate(list(ngrams(original_prob, n_range_prob)))
        ]
        scores = sorted(scores, reverse=True)
        prob   = np.zeros([len(sentence)])
        
        prediction = []
        for score, start, end in scores:
            if score > threshold:
                if np.sum(prob[start:end]) > 0:
                    continue
                prediction.append([score, sub_word, sub_sentence, start, end])
                # prediction.append([sub_word, '', [start, end]])
                prob[start:end] = original_prob[start:end]
        return sentence, prob, prediction

    def _preprocess(self, texts):
        return [" ".join(list(text)) for text in texts]

    def predict(self, texts: List[str]) -> List[str]:
        results = [self.predict_one_step(text) for text in tqdm(texts)]
        predictions = [d[0] for d in results]
        scores      = [d[1] for d in results]
        return predictions, scores

    def predict_one_step(self, text: str) -> List[str]:
        results = []
        sent_sylls, word2syllable, syllable2word, words = self.word_to_syllable(text)
        for i in range(len(self.entities_syllable)):
            sub_syll, sub_word = self.entities_syllable[i]
            # print(f'word: {sub_word}, sylls: {sub_syll}')
            sylls, prob, pred = self.sliding_window(sent_sylls, sub_syll, sub_word, 0.1)

            score = np.max(prob)
            results.append([score, pred])

        prob    = np.zeros([len(syllable2word)])
        results = sorted(results, reverse=True)

        final_results = []
        for score, prediction in results:
            for pred in prediction:
                score, sub_word, sub_syllable, start, end = pred
                if score <= 0.6:
                    break
                if np.sum(prob[start:end]) > 0:
                    continue
                prob[start:end] = score
                final_results.append([start, score, sub_word, '', [start, end]])
        final_results = sorted(final_results)

        _final_results = []
        final_scores   = []
        for start, score, sub_word, type, position in final_results:
            _final_results.append([sub_word, type, position])
            final_scores.append(score)
        return _final_results, final_scores


if __name__ == '__main__':
    entity_path = "/share/nas165/amian/experiments/speech/EntityCorrector/blists/aishell/test_1_entities.txt"
    text = "每日经济新闻记者杨建江南佳节六万"

    detector = PinyinDetector(entity_path)
    prediction = detector.predict([text])
    print(prediction)