from enum import Enum


class TokenType(Enum):
    META = 'meta'
    NONE = ''
    VERB = '動詞'
    AUXILARY_VERB = '助動詞'
    I_TYPE_ADJ = '形容詞'
    NA_TYPE_ADJ = '形容動詞'
    NA_TYPE_ADJ2 = '形状詞'
    NOUN = '名詞'
    PRONOUN = '代名詞'
    ADVERB = '副詞'
    CONJUCTION = '接続詞'
    INTERJECTION = '感動詞'
    PRENOMINAL = '連体詞'
    PREFIX = '接頭辞'
    SUFFIX = '接尾辞'
    PARTICLE = '助詞'
    PUNCTUATION = '補助記号'
    SYMBOL = '記号'
    BLANK = '空白'
