from typing import Dict, List, Tuple
from itertools import count

from fugashi import Tagger # type: ignore

from src.enums.language_enum import Languages
from src.utils.enhancers.text_enhancer import TextEnhancer
from src.utils.tokenizers.text_tokenizer import TextTokenizer


class SrtTextTokenizer(TextTokenizer):
    def tokenize(self, dataset_path: str, line_count: int, language: Languages, enhancement_variations: int) -> Tuple[List[List[str]], Dict[str, int]]:
        with open(dataset_path, 'r', encoding='utf8') as datasource:
            datasource_lines = datasource.readlines()
            filtered_lines = [
                line
                for line in datasource_lines
                if len(line) > 10
                and not line.startswith('http')
                and not line.endswith('.ja\n')
                and not '-->' in line
            ]

        if language == Languages.JAPANESE:
            dict_tagger = Tagger('-Owakati')

            token_collection = [
                self.__tokenize_sentence_jp(line, dict_tagger)
                for line in filtered_lines[:line_count]
            ]

            return self.__postprocess_tokens(token_collection, enhancement_variations), self.__get_dictionary_jp(filtered_lines[:line_count], dict_tagger)
        else:
            filtered_lines = ' '.join(filtered_lines).split('. ')

            token_collection = [
                self.__tokenize_sentence(line)
                for line in filtered_lines[:line_count]
            ]

            return self.__postprocess_tokens(token_collection, enhancement_variations), self.__get_dictionary(filtered_lines[:line_count])


    def __get_dictionary(self, lines: List[str]) -> Dict[str, int]:
        iter_count = count(3)
        result = {}
        entries = set()

        for line in lines:
            tags = self.__tokenize_sentence(line)

            for tag in tags:
                if tag in entries:
                    continue

                entries.add(tag)
                result[tag] = next(iter_count)

        result[''] = 0
        result['<|start|>'] = 1
        result['<|end|>'] = 2

        return result


    def __tokenize_sentence(self, sentence: str) -> List[str]:
        return [
            str(tag)
            for tag in sentence.replace(',', ' ,').replace('\n', '').split(' ')
        ]


    def __get_dictionary_jp(self, lines: List[str], dict_tagger: Tagger) -> Dict[str, int]:
        iter_count = count(3)
        result = {}
        entries = set()

        for line in lines:
            tags = self.__tokenize_sentence_jp(line, dict_tagger)

            for tag in tags:
                if tag in entries:
                    continue

                entries.add(tag)
                result[tag] = next(iter_count)

        result[''] = 0
        result['<|start|>'] = 1
        result['<|end|>'] = 2

        return result


    def __tokenize_sentence_jp(self, sentence: str, dict_tagger: Tagger = None) -> List[str]:
        if not dict_tagger:
            dict_tagger = Tagger('-Owakati')

        dict_tagger.parse(sentence)

        return [
            str(tag)
            for tag in dict_tagger(sentence)
        ]


    def __postprocess_tokens(self, token_collection: List[List[str]], variations: int) -> List[List[str]]:
        max_len = max(len(token_list) for token_list in token_collection)

        if variations > 0:
            enhanced_collection = TextEnhancer().randomize_token_sequences(token_collection, variations)

            for token_list in enhanced_collection:
                while len(token_list) < max_len:
                    token_list.append('')

            return enhanced_collection

        for token_list in token_collection:
                token_list.insert(0, '<|start|>')
                token_list.append('<|end|>')

                while len(token_list) < max_len + 2:
                    token_list.append('')

        return token_collection
