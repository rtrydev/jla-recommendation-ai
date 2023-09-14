from typing import Dict, List, Tuple
from itertools import count

from fugashi import Tagger # type: ignore

from src.utils.tokenizers.text_tokenizer import TextTokenizer


class SrtTextTokenizer(TextTokenizer):
    def tokenize(self, dataset_path: str, line_count: int) -> Tuple[List[List[str]], Dict[str, int]]:
        with open(dataset_path, 'r', encoding='utf8') as datasource:
            datasource_lines = datasource.readlines()
            filtered_lines = [
                line
                for line in datasource_lines
                if len(line) > 10
                and not line.startswith('http')
                and not line.endswith('.ja\n')
            ]

        dict_tagger = Tagger('-Owakati')

        token_collection = [
            self.__tokenize_sentence(line, dict_tagger)
            for line in filtered_lines[:line_count]
        ]

        return self.__postprocess_tokens(token_collection), self.__get_dictionary(filtered_lines[:line_count], dict_tagger)


    def __get_dictionary(self, lines: List[str], dict_tagger: Tagger) -> Dict[str, int]:
        iter_count = count(2)
        result = {}
        entries = set()

        for line in lines:
            dict_tagger.parse(line)

            tags = [
                str(tag)
                for tag in dict_tagger(line)
            ]

            for tag in tags:
                if tag in entries:
                    continue

                entries.add(tag)
                result[tag] = next(iter_count)

        result['<|end|>'] = 1
        result[''] = 0

        return result


    def __tokenize_sentence(self, sentence: str, dict_tagger: Tagger = None) -> List[str]:
        if not dict_tagger:
            dict_tagger = Tagger('-Owakati')

        dict_tagger.parse(sentence)

        return [
            str(tag)
            for tag in dict_tagger(sentence)
        ]


    def __postprocess_tokens(self, token_collection: List[List[str]]) -> List[List[str]]:
        max_len = max(len(token_list) for token_list in token_collection)

        for token_list in token_collection:
            token_list.append('<|end|>')

            while len(token_list) < max_len + 1:
                token_list.append('')

        return token_collection
