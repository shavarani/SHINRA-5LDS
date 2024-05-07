"""
This script creates a unified tokenizer for SHINRA-5LDS dataset
"""
from tqdm import tqdm
from dataset import SHINRA5LDS
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

def output_dataset():
    for lang in ['ja', 'en', 'fr', 'de', 'fa']:
        with open(f"path_to_{lang}_corpus.txt", "w", encoding='utf-8') as f:
            for article, _ in tqdm(SHINRA5LDS('SHINRA-5LDS.zip', lang)):
                f.write(article.title+'\n')
                for sentence in article.content.split('\n'):
                    if sentence.strip():
                        f.write(sentence+'\n')
                f.write(article.url+' '+article.page_id+'\n')
                if article.links:
                    for link in article.links:
                        f.write(link+'\n')
                for sentence in article.summary.split('\n'):
                    if sentence.strip():
                        f.write(sentence+'\n')
                if article.categories:
                    for category in article.categories:
                        f.write(category+'\n')

def train_tokenizer():
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(["<s>", "<pad>", "</s>", "<unk>", "<mask>", "<lang_jp>", "<lang_en>", "<lang_fr>", "<lang_de>", "<lang_fa>"])
    language_codes = {"ja": "<lang_ja>", "en": "<lang_en>", "fr": "<lang_fr>", "de": "<lang_de>", "fa": "<lang_fa>"}
    def add_language_identifier(text, lang_code):
        return language_codes[lang_code] + " " + text
    trainer = trainers.BpeTrainer(special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>", "<lang_jp>", "<lang_en>", "<lang_fr>", "<lang_de>", "<lang_fa>"])
    tokenizer.train_from_iterator((add_language_identifier(line.strip(), lang) for lang in ["ja", "en", "fr", "de", "fa"]
                                   for line in open(f"path_to_{lang}_corpus.txt", "r", encoding="utf-8")), trainer=trainer)
    tokenizer.save("multilingual_tokenizer_with_lang.json")

if __name__ == "__main__":
    output_dataset()
    train_tokenizer()