from dataclasses import asdict
import json
from typing import Dict, List, Set, Tuple
from itertools import count

from fugashi import Tagger # type: ignore

from src.enums.language_enum import Languages
from src.enums.token_type_enum import TokenType
from src.models.token_data import TokenData
from src.utils.enhancers.text_enhancer import TextEnhancer
from src.utils.factories.asdict_factory import enum_asdict_factory
from src.utils.tokenizers.text_tokenizer import TextTokenizer
from src.utils.translators.kana_translator import translate_katakana


class SrtTextTokenizer(TextTokenizer):
    def tokenize(self, dataset_path: str, line_count: int, language: Languages, enhancement_variations: int, neighborhood_size: int) -> Tuple[List[List[str]], Dict[str, TokenData]]:
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
                self.__tokenize_sentence(line, dict_tagger)
                for line in filtered_lines[:line_count]
            ]

            return self.__postprocess_tokens(token_collection, enhancement_variations, neighborhood_size), self.__get_dictionary(filtered_lines[:line_count], dict_tagger)
        else:
            filtered_lines = ' '.join(filtered_lines).split('. ')

            token_collection = [
                self.__tokenize_sentence(line)
                for line in filtered_lines[:line_count]
            ]

            return self.__postprocess_tokens(token_collection, enhancement_variations, neighborhood_size), self.__get_dictionary(filtered_lines[:line_count])

    def save_tokens(self, token_dict: Dict[str, TokenData], file_name: str) -> None:
        dumped_dict = json.dumps({
            key: asdict(value, dict_factory=enum_asdict_factory)
            for key, value in token_dict.items()
        })

        with open(file_name, 'wb') as dumpfile:
            dumpfile.write(dumped_dict.encode('utf8'))

    def __get_dictionary(self, lines: List[str], dict_tagger: Tagger = None) -> Dict[str, TokenData]:
        iter_count = count(3)
        result: Dict[str, TokenData] = {}
        entries: Set[str] = set()

        for line in lines:
            tags = self.__get_tags(line, dict_tagger)

            for tag in tags:
                if tag.token in entries:
                    continue

                entries.add(tag.token)
                tag.token_id = next(iter_count)

                result[tag.token] = tag

        result[''] = TokenData(
            token_id=0,
            lemma_id=None,
            token='',
            infinitive=None,
            reading=None,
            token_type=TokenType.META
        )
        result['<|start|>'] = TokenData(
            token_id=1,
            lemma_id=None,
            token='<|start|>',
            infinitive=None,
            reading=None,
            token_type=TokenType.META
        )
        result['<|end|>'] = TokenData(
            token_id=2,
            lemma_id=None,
            token='<|end|>',
            infinitive=None,
            reading=None,
            token_type=TokenType.META
        )

        return result

    def __tokenize_sentence(self, sentence: str, dict_tagger: Tagger = None) -> List[str]:
        if not dict_tagger:
            return [
                str(tag)
                for tag in sentence.replace(',', ' ,').replace('\n', '').split(' ')
            ]

        dict_tagger.parse(sentence)

        return [
            str(tag)
            for tag in dict_tagger(sentence)
        ]

    def __get_tags(self, line: str, dict_tagger: Tagger = None) -> List[TokenData]:
        if not dict_tagger:
            return [
                TokenData(
                    token_id=-1,
                    lemma_id=None,
                    token=str(tag),
                    infinitive=str(tag),
                    reading=None,
                    token_type=TokenType('')
                )
                for tag in line.replace(',', ' ,').replace('\n', '').split(' ')
            ]
        else:
            return [
                TokenData(
                    token_id=-1,
                    lemma_id=str(tag.feature.lemma_id),
                    token=str(tag),
                    infinitive=str(tag.feature.lemma),
                    reading=translate_katakana(str(tag.feature.kanaBase)),
                    token_type=TokenType(tag.feature.pos1)
                )
                for tag in dict_tagger(line)
            ]

    def __postprocess_tokens(self, token_collection: List[List[str]], variations: int, neighborhood_size: int) -> List[List[str]]:
        enhanced_collection = None

        if neighborhood_size > 0:
            enhanced_collection = TextEnhancer().enhance_neighborhood(token_collection, neighborhood_size)

        max_len = max(len(token_list) for token_list in (token_collection if enhanced_collection is None else enhanced_collection))

        if variations > 0:
            enhanced_collection = TextEnhancer().randomize_token_sequences(token_collection if enhanced_collection is None else enhanced_collection, variations)

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
