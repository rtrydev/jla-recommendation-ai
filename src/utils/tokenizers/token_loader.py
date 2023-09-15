import json
from typing import Dict

from src.models.token_data import TokenData


class TokenLoader:
    def load(self, path: str) -> Dict[str, TokenData]:
        with open(path, 'r', encoding='utf8') as tokendump:
            dumped_dict = tokendump.read()

        dict_json = json.loads(dumped_dict)

        return {
            key: TokenData(
                token_id=value.get('token_id'),
                lemma_id=value.get('lemma_id'),
                token=value.get('token'),
                infinitive=value.get('infinitive'),
                reading=value.get('reading'),
                token_type=value.get('token_type')
            )
            for key, value in dict_json.items()
        }
