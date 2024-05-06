"""
    Example data reader for the annotated SHINRA-5LDS
    Created May 2, 2024.
    Copyright Hassan S. Shavarani
"""
import os
import json
import datetime
from zipfile import ZipFile
from torch.hub import download_url_to_file
from collections import namedtuple

ENELabel = namedtuple('ENELabel', ['level_id', 'name']) 


class ENEAnnotation:
    def __init__(self, ann):
        self.ene_id = ann['ene_id']
        self.representative_name = ann['representative_name']
        self._id = int(ann['_id'])
        self.original_level = ann['original_level']
        self.values = {x['level']: ENELabel(x['level_id'], x['name']) for x in ann['entities']}

class SHINRAAnnotation:
    def __init__(self, annotation):
        self.lang = annotation['lang'] # this value is always equal to 'ja' since all original annotations have been assigned in Japanese
        self.annotator = annotation['annotator']
        self.annotation_date = datetime.datetime.strptime(annotation['date'], '%Y-%m-%d %H:%M:%S')
        self.raw_flag = annotation['raw_flag']
        self.type = annotation['type']
        self.annotation_confidence = annotation['probability_score']
        self.labels = ENEAnnotation(annotation['ene_id'])


class SHINRAArticle:
    def __init__(self, article):
        self.title = article['title']
        self.content = article['content']
        self.url = article['url']
        self.page_id = article['page_id']
        self.links = article['links']
        self.summary = article['summary']
        self.categories = article['categories']

class SHINRARecord:
    def __init__(self, record):
        self.annotation_id = record['annotation_id']
        self.ene_level_ids = {
            "0": record['ene_level_ids']['level0'],
            "1": record['ene_level_ids']['level1'],
            "2": record['ene_level_ids']['level2'],
            "3": record['ene_level_ids']['level3']
        }
        self.ene_label_verified = record['ene_label_verified']
        self.annotations = [SHINRAAnnotation(x) for x in record['annotations'] if x['annotator'] == 'HAND']
        self.articles = {x['language']: SHINRAArticle(x) for x in record['annotation_articles']}
    
    @property
    def latest_annotation(self):
        return max(self.annotations, key=lambda x: x.annotation_date)

class SHINRA5LDS:
    """
    Size of the SHINRA-5LDS dataset (considering only the hand annotated records):
        max annotation count for all 5 languages: 5
        ja: 118635 records, 1.0357 annotations per article, 742.5 articles per class, 164 classes
        en:  52445 records, 1.0357 annotations per article, 339.9 articles per class, 159 classes
        fr:  34432 records, 1.0346 annotations per article, 227.2 articles per class, 156 classes
        de:  29808 records, 1.0306 annotations per article, 198.6 articles per class, 154 classes
        fa:  14058 records, 1.0335 annotations per article,  97.7 articles per class, 148 classes
    cross validation: 80/10/10
    """
    def __init__(self, zip_file_path, lang):
        self.zip_file_path = zip_file_path
        self.data = None
        self.lang = lang
        if "SHINRA-5LDS.zip" not in os.listdir():
            download_url_to_file("https://huggingface.co/datasets/sshavara/SHINRA-5LDS/resolve/main/SHINRA-5LDS.zip?download=true", zip_file_path, progress=True)

    def _load_data(self):
        zip_ref = ZipFile(self.zip_file_path, 'r')
        self.data = zip_ref.open('data.jsonl')

    def __iter__(self):
        if self.data is None or len(self.data) == 0:
            self._load_data()
        return self

    def __next__(self):
        try:
            while True:
                next_line = self.data.readline()
                if not next_line:
                    self.data.close()
                    self.data = None
                    raise IndexError
                record = SHINRARecord(json.loads(next_line))
                if not record.annotations: # the annotation is machine generated
                    continue
                if self.lang in record.articles:
                    break
            annotations = [[x.labels.values[i].name for x in record.annotations] for i in range(4)] # if x.representative_name != 'IGNORED'
            return record.articles[self.lang], annotations
        except IndexError:
            raise StopIteration
        except Exception as e:
            print(f"Error: {e}")
            return self.__next__()
