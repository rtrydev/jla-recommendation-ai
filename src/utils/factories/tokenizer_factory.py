from src.utils.tokenizers.srt_text_tokenizer import SrtTextTokenizer
from src.utils.tokenizers.text_tokenizer import TextTokenizer


def create_tokenizer(dataset: str) -> TextTokenizer:
    if dataset.endswith('.srt'):
        return SrtTextTokenizer()
    if dataset.endswith('.txt'):
        return SrtTextTokenizer()

    raise NotImplementedError('Unsupported file extension!')
