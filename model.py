from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score

from dataset import SHINRA5LDS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ene_vocab = {
    0 : {'IGNORED': 0, 'CONCEPT': 1, 'Numex': 2, 'Time_TOP': 3, 'Name': 4},
    1:  {'IGNORED': 0, 'Color': 1, 'Product': 2, 'Countx': 3, 'Location': 4, 'Organization': 5, 'Timex': 6, 'Person': 7, 'Facility': 8, 'Natural_Object': 9, 'Latitude_Longtitude': 10, 'Age': 11, 'Numex_Other': 12, 'God': 13, 'Event': 14, 'Ordinal_Number': 15, 'Periodx': 16, 'Percent': 17, 'CONCEPT': 18, 'Disease': 19, 'Name_Other': 20, 'Measurement': 21},
    2:  {'IGNORED': 0, 'Event_Other': 1, 'Natural_Object_Other': 2, 'GOE': 3, 'Character': 4, 'International_Organization': 5, 'Period_Day': 6, 'Geological_Region': 7, 'Occasion': 8, 'Printing': 9, 'Periodx_Other': 10, 'Numex_Other': 11, 'Nature_Color': 12, 'Food': 13, 'CONCEPT': 14, 'Clothing': 15, 'Facility_Other': 16, 'God': 17, 'Vehicle': 18, 'Weapon': 19, 'Sports_Organization': 20, 'Period_Year': 21, 'Award': 22, 'Latitude_Longtitude': 23, 'Time': 24, 'Show_Organization': 25, 'Living_Thing_Part': 26, 'Person': 27, 'Service': 28, 'Organization_Other': 29, 'Decoration': 30, 'Living_Thing': 31, 'Disease_Other': 32, 'Astral_Body': 33, 'Corporation': 34, 'Language': 35, 'Product_Other': 36, 'Compound': 37, 'Spa': 38, 'Money_Form': 39, 'Art': 40, 'Natural_Phenomenon': 41, 'Ethnic_Group': 42, 'Date': 43, 'Animal_Disease': 44, 'Facility_Part': 45, 'Color_Other': 46, 'Seismic_Intensity': 47, 'Archaeological_Place': 48, 'Political_Organization': 49, 'Region': 50, 'Era': 51, 'ID_Number': 52, 'Address': 53, 'Doctrine_Method': 54, 'Name_Other': 55, 'GPE': 56, 'Line': 57, 'Material': 58, 'N_Person': 59, 'Unit': 60, 'Mineral': 61, 'Percent': 62, 'Measurement_Other': 63, 'Family': 64, 'Title': 65, 'Age': 66, 'Offence': 67, 'Ordinal_Number': 68, 'Incident': 69, 'Element': 70, 'Drug': 71, 'Location_Other': 72, 'Rule': 73},
    3:  {'IGNORED': 0, 'Incident_Other': 1, 'Station': 2, 'CONCEPT': 3, 'Aircraft': 4, 'Government': 5, 'Company_Group': 6, 'Nature_Color': 7, 'Doctrine_Method_Other': 8, 'Printing_Other': 9, 'Public_Institution': 10, 'Element': 11, 'Drug': 12, 'ID_Number': 13, 'Market': 14, 'Region_Other': 15, 'Religion': 16, 'Broadcast_Program': 17, 'Date': 18, 'Geological_Region_Other': 19, 'Mountain': 20, 'Vehicle_Other': 21, 'Train': 22, 'Period_Day': 23, 'Natural_Object_Other': 24, 'Cabinet': 25, 'Music': 26, 'Fish': 27, 'Natural_Phenomenon_Other': 28, 'Ship': 29, 'Seismic_Intensity': 30, 'Sea': 31, 'Material': 32, 'Theory': 33, 'Bridge': 34, 'Airport': 35, 'Facility_Other': 36, 'Political_Party': 37, 'Name_Other': 38, 'Railroad': 39, 'Show_Organization': 40, 'Spaceship': 41, 'Movement': 42, 'Person': 43, 'Product_Other': 44, 'Reptile': 45, 'Canal': 46, 'Spa': 47, 'Style': 48, 'Rule_Other': 49, 'Disease_Other': 50, 'GPE_Other': 51, 'Earthquake': 52, 'Title_Other': 53, 'Museum': 54, 'Art_Other': 55, 'Province': 56, 'Mammal': 57, 'Organization_Other': 58, 'Animal_Part': 59, 'Currency': 60, 'Position_Vocation': 61, 'Mineral': 62, 'Flora': 63, 'Park': 64, 'National_Language': 65, 'River': 66, 'Amphibia': 67, 'Ethnic_Group_Other': 68, 'City': 69, 'Academic': 70, 'Bird': 71, 'Time': 72, 'Living_Thing_Part_Other': 73, 'Magazine': 74, 'Mollusc_Arthropod': 75, 'Compound': 76, 'Port': 77, 'Road': 78, 'County': 79, 'Award': 80, 'Car': 81, 'Continental_Region': 82, 'Book': 83, 'Periodx_Other': 84, 'Flora_Part': 85, 'Address_Other': 86, 'Tunnel': 87, 'Military': 88, 'Numex_Other': 89, 'Theater': 90, 'Latitude_Longtitude': 91, 'Language_Other': 92, 'Archaeological_Place_Other': 93, 'International_Organization': 94, 'Event_Other': 95, 'GOE_Other': 96, 'Research_Institute': 97, 'Clothing': 98, 'Plan': 99, 'Offence': 100, 'Percent': 101, 'Sports_Organization_Other': 102, 'Location_Other': 103, 'Service': 104, 'Ordinal_Number': 105, 'Domestic_Region': 106, 'Character': 107, 'Zoo': 108, 'Astral_Body_Other': 109, 'Star': 110, 'Decoration': 111, 'Animal_Disease': 112, 'Amusement_Park': 113, 'Movie': 114, 'Conference': 115, 'Measurement_Other': 116, 'Company': 117, 'Water_Route': 118, 'Worship_Place': 119, 'Occasion_Other': 120, 'Pro_Sports_Organization': 121, 'Game': 122, 'Sport': 123, 'Natural_Disaster': 124, 'Dish': 125, 'Constellation': 126, 'Corporation_Other': 127, 'Planet': 128, 'Color_Other': 129, 'Age': 130, 'Picture': 131, 'N_Person': 132, 'Facility_Part': 133, 'Newspaper': 134, 'Insect': 135, 'Sports_League': 136, 'Living_Thing_Other': 137, 'Food_Other': 138, 'Fungus': 139, 'Treaty': 140, 'Lake': 141, 'Car_Stop': 142, 'Island': 143, 'Culture': 144, 'Political_Organization_Other': 145, 'Country': 146, 'School': 147, 'Unit_Other': 148, 'Religious_Festival': 149, 'Line_Other': 150, 'God': 151, 'Era': 152, 'Weapon': 153, 'Show': 154, 'War': 155, 'Family': 156, 'Period_Year': 157, 'Money_Form': 158, 'Nationality': 159, 'Bay': 160, 'Sports_Facility': 161, 'Tumulus': 162, 'Law': 163}
}


class TextClassificationDataset(Dataset):
    def __init__(self, tokenizer, max_length, lang):
        self.dataset = list(tqdm(SHINRA5LDS('SHINRA-5LDS.zip', lang)))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lang = lang

    def __len__(self):
        if self.lang == 'ja':
            return 118635
        elif self.lang == 'en':
            return 52445
        elif self.lang == 'fr':
            return 34432
        elif self.lang == 'de':
            return 29808
        elif self.lang == 'fa':
            return 14058
        else:
            raise ValueError(f'Invalid language: {self.lang}')

    def __getitem__(self, idx):
        article, annotations = self.dataset[idx]
        inputs = self.tokenizer.encode_plus(
            article.title + ' ' + article.content.split("\n")[0], # takse this as a configurable parameter
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        level_annotation_ids = [torch.zeros(len(ene_vocab[0])).to(device), 
                                torch.zeros(len(ene_vocab[1])).to(device), 
                                torch.zeros(len(ene_vocab[2])).to(device), 
                                torch.zeros(len(ene_vocab[3])).to(device)]
        for label_level in range(4):
            level_annotations = [ene_vocab[label_level][x] for x in annotations[label_level]]
            level_annotation_ids[label_level][level_annotations] = 1
        input_ids = inputs['input_ids'].squeeze().to(device)
        attention_mask = inputs['attention_mask'].squeeze().to(device)
        return input_ids, attention_mask, *level_annotation_ids
    
class Classifier(nn.Module):
    def __init__(self, model_name, freeze_encoder=False, threshold=0.5, affine_mid_layer_size=256):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        if freeze_encoder:
            for param in self.transformer.parameters():
                param.requires_grad = False
        self.affine_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.transformer.config.hidden_size, affine_mid_layer_size),
                nn.ReLU(),
                nn.Linear(affine_mid_layer_size, len(ene_vocab[i])),
                # nn.Linear(self.transformer.config.hidden_size, len(ene_vocab[i])),
                # nn.Threshold(threshold, 0)
            ) for i in range(4)
        ])
        self.threshold = threshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]
        x = [layer(pooled_output) for layer in self.affine_layers]
        return x

    def inference(self, input_ids, attention_mask):
        with torch.no_grad():
            return [(self.sigmoid(x) > self.threshold).float() for x in self.forward(input_ids, attention_mask)]

def cross_valid(model_name = 'roberta-base', max_length=512, lang='en', k_folds=10, lr=1e-5, batch_size=32, num_epochs=10, membership_threshold=0.5, freeze_encoder=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print('Pre-loading dataset ...')
    dataset = TextClassificationDataset(tokenizer, max_length, lang)
    classifier = Classifier(model_name, freeze_encoder=freeze_encoder, threshold=membership_threshold).to(device)
    skf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    accuracies = [[] for _ in range(4)]
    precisions = [[] for _ in range(4)]
    recalls = [[] for _ in range(4)]
    f1_scores = [[] for _ in range(4)]

    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)))):
        print(f'Fold {fold+1}/{k_folds} ...')
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            total_loss = 0
            total_count = 0
            classifier.train()
            train_iter = tqdm(train_loader)
            for input_ids, attention_mask, *level_annotation_ids in train_iter:
                optimizer.zero_grad()
                loss = sum([criterion(output, labels) for output, labels in zip(classifier(input_ids, attention_mask), level_annotation_ids)])
                total_loss += loss.item()
                total_count += input_ids.size(0)
                loss.backward()
                optimizer.step()
                train_iter.set_description(f'Loss: {total_loss / float(total_count):.5f}')
                

        classifier.eval()
        correct = [0, 0, 0, 0]
        total = [0, 0, 0, 0]
        level_labels = [[] for _ in range(4)]
        level_predicted = [[] for _ in range(4)]
        print('Validation at the end of the fold ...')
        with torch.no_grad():
            for val_id, (input_ids, attention_mask, *level_annotation_ids) in enumerate(val_loader):
                for level_id, (output, labels) in enumerate(zip(classifier.inference(input_ids, attention_mask), level_annotation_ids)):
                    # output = output > membership_threshold
                    total[level_id] += labels.count_nonzero().item()
                    predicted = (output * labels).bool()
                    correct[level_id] += torch.sum(predicted).item()
                    level_labels[level_id].extend(labels.view(-1).cpu().numpy())
                    level_predicted[level_id].extend(output.view(-1).cpu().numpy())

        print('Evaluation results at the end of the fold ...')
        for level_id in range(4):
            accuracy = correct[level_id] / total[level_id]
            precision = precision_score(level_labels[level_id], level_predicted[level_id], average='macro', zero_division=0)
            recall = recall_score(level_labels[level_id], level_predicted[level_id], average='macro', zero_division=0)
            f1 = f1_score(level_labels[level_id], level_predicted[level_id], average='macro', zero_division=0)

            print(f'Level {level_id}:')
            print(f'\tAccuracy: {accuracy:.2f}')
            print(f'\tPrecision: {precision:.2f}')
            print(f'\tRecall: {recall:.2f}')
            print(f'\tF1-score: {f1:.2f}')

            accuracies[level_id].append(accuracy)
            precisions[level_id].append(precision)
            recalls[level_id].append(recall)
            f1_scores[level_id].append(f1)

    for level_id in range(4):
        avg_accuracy = np.mean(accuracies[level_id]) * 100
        std_accuracy = np.std(accuracies[level_id]) * 100
        avg_precision = np.mean(precisions[level_id]) * 100
        std_precision = np.std(precisions[level_id]) * 100
        avg_recall = np.mean(recalls[level_id]) * 100
        std_recall = np.std(recalls[level_id]) * 100
        avg_f1 = np.mean(f1_scores[level_id]) * 100
        std_f1 = np.std(f1_scores[level_id]) * 100
        print('='*120)
        print(f'Level {level_id}:')
        print(f'\tAverage Accuracy: {avg_accuracy:.2f}, Std: {std_accuracy:.2f}')
        print(f'\tAverage Precision: {avg_precision:.2f}, Std: {std_precision:.2f}')
        print(f'\tAverage Recall: {avg_recall:.2f}, Std: {std_recall:.2f}')
        print(f'\tAverage F1-score: {avg_f1:.2f}, Std: {std_f1:.2f}')
        print('='*120)



if __name__ == '__main__':
    cross_valid(model_name = 'roberta-base', max_length=512, lang='en', k_folds=5, lr=1e-5, batch_size=32, num_epochs=5, membership_threshold=0.5, freeze_encoder=False)