import numpy as np
import pandas as pd

import sys
import re
from xml.etree import ElementTree as ET

#объявление классов
class TextStats():

    # вспомогательная функция для парсинга тегов fb2
    def parse_p(self, node):
        words = []
        if list(node):

            # Лучше это писать через try ... except, но я боюсь, что try .. except TypeError
            # может замаскировать непредвиденные ошибки
            if node.text:
                words += [[t, 0] for t in re.split('\s+', node.text)]

            for sub in list(node):

                if sub.tag == 'a':
                    pass
                elif sub.tag == 'emphasis':
                    # выделенный текст
                    words += [[t, 1] for t in re.split('\s+', sub.text)]

                else:
                    print(sub.tag)
                    print(sub.text)

                # то же самое, что выше: читается лучше с try, но с точки зрения fail fast лучше так
                if sub.tail:
                    words += [[t, 0] for t in re.split('\s+', sub.tail)]

        else:
            words += [[t, 0] for t in re.split('\s+', node.text)]
        return words

    # рекурсивно обходим вложенные разделы в fb2-файле, достаём названия глав
    def word_table(self, node, title):
        # print('node = {}'.format(node.tag))
        word_list = []
        table_list = []
        subtitle_text = None
        words = pd.DataFrame(data={'word': [],
                                   'chapter': [],
                                   'is_emphasized': []})
        for el in list(node):
            if el.tag == 'section':
                subtitle = el.find('title/p')

                try:
                    subtitle_text = subtitle.text
                except AttributeError:
                    pass
                if title:
                    subtitle_text = title + '. ' + subtitle_text
                else:
                    subtitle_text = subtitle_text

                # print(subtitle_text)

                table_list += self.word_table(el, subtitle_text)
            elif el.tag == 'p':
                word_list += self.parse_p(el)


            elif el.tag == 'poem':
                for v_tag in el.find('./stanza/v'):
                    word_list += [[w, 0] for w in re.split('\s+', v_tag.text)]
            elif el.tag == 'cite':
                if el.find('./p'):
                    for p_tag in el.find('./p'):
                        word_list += self.parse_p(p_tag)
                else:
                    try:
                        for subtitle_tag in el.find('./subtitle'):
                            word_list += [[w, 0] for w in re.split('\s+', subtitle_tag.text)]
                    except TypeError:
                        # редкие косяки (до 15 на книгу), не успеваю обработать
                        pass
        if word_list:
            res = pd.DataFrame(data=np.array(word_list), columns=['word', 'is_emphasized'])
            res['chapter'] = title
            return [res]
        else:
            return table_list

    def __init__(self, path):

        self.name = path.replace('.fb2', '')

        # открываем файл, парсим xml
        tree = ET.parse(path)
        body = [child for child in tree.findall('./') if 'body' in child.tag and not child.attrib][0]

        # собираем и объединяем главы в единый df 
        self.raw_table = pd.concat(self.word_table(body, None)).reset_index(drop=True)
        self.words_clean = None
        self.sentences_clean = None

    def clean_data(self):
        end_sentence = re.compile('[\.\!\?»…]+')
        start_sentence = re.compile('«')
        dialog_break = '–'

        # удаляем переносы строк
        self.raw_table.drop(self.raw_table[self.raw_table['word'] == ''].index, inplace=True)

        # выделим и пронумеруем предложения
        self.raw_table['sentence_n'] = (self.raw_table['word']
                                   .apply(lambda x: 1 if re.search(end_sentence, x) else 0)
                                   .shift(1)
                                   .fillna(0))
        self.raw_table.loc[0, 'sentence_n'] = 1
        self.raw_table.reset_index(drop=True, inplace=True)
        self.raw_table.loc[self.raw_table[(self.raw_table['word'] == dialog_break) &
                                (self.raw_table['sentence_n'] > 0)].index + 1, 'sentence_n'] += 1

        # считаем предложения из диалогов
        self.raw_table['dialog'] = 0
        self.raw_table.loc[self.raw_table[(self.raw_table['word'] == dialog_break) &
                                (self.raw_table['sentence_n'] > 0)].index + 1, 'dialog'] = 1

        # подчищаем теперь уже лишние тире в начале предложений
        self.raw_table.drop(self.raw_table[self.raw_table['word'] == dialog_break].index, inplace=True)
        self.raw_table.reset_index(drop=True, inplace=True)

        self.raw_table['sentence'] = self.raw_table['sentence_n'].cumsum()

        # посчитаем запятые и точки с запятыми
        self.raw_table['comma'] = self.raw_table['word'].str.contains(',').astype(int)
        self.raw_table['semicolon'] = self.raw_table['word'].str.contains(';').astype(int)

        # посчитаем знаки экспрессии
        self.raw_table['exclamation'] = self.raw_table['word'].str.contains('!').astype(int)
        self.raw_table['q_mark'] = self.raw_table['word'].str.contains('\?').astype(int)
        self.raw_table['ddd'] = self.raw_table['word'].str.contains('…|\?\.\.|!\.\.').astype(int)

        # добавим номера глав
        self.raw_table['chapter_n'] = ((self.raw_table['chapter'].shift(1) != self.raw_table['chapter'])
                                  .astype(int)
                                  .cumsum())
        # двойные знаки пунктуации разделялись на 2 строчки. Когда мы посчитали все знаки,
        # можно удалить такие строки из 1го знака пунктуации
        self.raw_table = (self.raw_table.drop(self.raw_table[self.raw_table['word'].str.match(r'\W$')].index)
                     .reset_index(drop=True))

        #
        # пунктуацию посчитали, можно убирать 
        self.raw_table['word'] = self.raw_table['word'].str.replace(r'(^\W+)|(\W+$)', '')

        # находим имена собственные. 
        # Re плох с заглавными буквами в юникоде, я проявляю смекалку, чтобы не писать сложных выражений 
        self.raw_table['is_capital'] = (self.raw_table['word'].str.slice(stop=1) ==
                                   self.raw_table['word'].str.slice(stop=1).str.upper())
        self.raw_table['is_proper'] = (self.raw_table['is_capital'] & (self.raw_table['sentence_n'] == 0))

        # теперь можно нормализовать регистр
        self.raw_table['word'] = self.raw_table['word'].str.lower()

        # выделим слова русского языка
        self.raw_table['is_russian'] = self.raw_table['word'].str.contains(r'[А-Яа-яЁё]')

        # некоторые нерусские слова - цифры
        self.raw_table['is_digit'] = self.raw_table['word'].str.isdigit()

        # длина слов и количество гласных
        self.raw_table['length'] = self.raw_table['word'].apply(len)
        self.raw_table['vowels'] = self.raw_table['word'].apply(lambda x: len(re.findall(r'[аоиеяюыэё]', x)))

        # насладимся результатом
        self.words_clean = self.raw_table[['word', 'sentence', 'chapter_n', 'length', 'is_digit', 'is_russian',
                                      'is_proper', 'is_emphasized', 'vowels']]

        # отдельно статистика по предложениям
        self.sentences_clean = (self.raw_table
                                .groupby(['sentence'])['comma', 'semicolon', 'exclamation', 'q_mark', 'ddd']
                                .apply(np.sum)
                                .merge(self.raw_table
                                       .groupby(['sentence']).agg({'word': 'count', 'length': 'mean'}),
                                       left_index=True, right_index=True)
                                )

    # Это не очень красивый код, надо было бы сделать отдельный класс для статистик...
    def calc_basic_stats(self):
        self.basic_stats = {}
        self.basic_stats['words_total'] = self.words_clean.shape[0]
        self.basic_stats['words_rus'] = self.words_clean[(~(self.words_clean['is_digit'])
                                      & self.words_clean['is_russian'])].shape[0] / self.basic_stats['words_total']
        self.basic_stats['words_in_sentence'] = self.sentences_clean['word'].mean()
        self.basic_stats['sentences_total'] = self.sentences_clean.shape[0]
        self.basic_stats['sentences_per_chapter'] = (self.words_clean[['sentence', 'chapter_n']]
                                      .drop_duplicates()
                                      .groupby('chapter_n')
                                      .agg('count')['sentence']
                                      .mean())
        self.basic_stats['words_per_chapter'] = self.words_clean.groupby('chapter_n').count()['word'].mean()
        self.basic_stats['chapters_total'] = self.words_clean['chapter_n'].drop_duplicates().shape[0]
        self.basic_stats['punctuation_per_sentence'] = ((self.sentences_clean['comma'] +
                                         self.sentences_clean['semicolon']).sum() 
                                         / self.basic_stats['sentences_total'])
        self.basic_stats['sentiment_per_sentence'] = ((self.sentences_clean['exclamation']
                                                      + self.sentences_clean['q_mark']
                                                      + self.sentences_clean['ddd']).sum()
                                                      / self.basic_stats['sentences_total'])
        self.basic_stats['exclamations'] = ((self.sentences_clean['exclamation']).sum()
                                            / self.basic_stats['sentences_total'])
        self.basic_stats['q_marks'] = (self.sentences_clean['q_mark']).sum() / self.basic_stats['sentences_total']
        self.basic_stats['ddd'] = (self.sentences_clean['ddd']).sum() / self.basic_stats['sentences_total']

    def calc_advanced_stats(self):
        return
#объявление процедур



# тело скрипта
# Вызывать с 2мя атрибутами:
# file1: путь к файлу с текстом 1 в формате fb2
# file2: путь к файлу с текстом 2 в формате fb2
if __name__ == "__main__":
    # читаем параметры
    file1 = "p+n.fb2"
    file2 = "v+m.fb2"
    # file1 = None
    # file2 = None
    # try:
    #     file1 = sys.argv[1]
    # except IndexError:
    #     print("Укажите 2 файла для сравнения (сейчас указано 0)")
    #
    # if file1:
    #     try:
    #         file2 = sys.argv[2]
    #     except IndexError:
    #         print("Укажите 2 файла для сравнения (сейчас указан 1)")
    #
    # if not(file1 and file2):
    #     sys.exit(0)
    comp1 = TextStats(file1)
    comp2 = TextStats(file2)

    comp1.clean_data()
    comp2.clean_data()

    comp1.calc_basic_stats()
    comp2.calc_basic_stats()

    pd.DataFrame(data={metric: [comp1.basic_stats[metric], comp2.basic_stats[metric]]
                       for metric in comp1.basic_stats.keys()},
                 index=[comp1.name, comp2.name],
                 columns=comp1.basic_stats.keys()).to_excel('basic_report.xlsx')


