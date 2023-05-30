import io
import os
import time
import jieba

from src.utils import read_file
from src.utils import write_file
from src.utils import write_json

OUTPUT_DIR = './dump'

def get_transcript_text(transcript_path):
    with open(transcript_path) as f:
        return [line.strip().split(" ", 1)[1].lower() for line in f]

def get_transcripts(dataset_path):
    transcript_paths = dataset_path.glob("*/*/*.trans.txt")
    merged_transcripts = []
    for path in transcript_paths:
        merged_transcripts += get_transcript_text(path)
    return merged_transcripts
    
def word_frequency(text, freqlist):
    for word in text.split(' '):
        freqlist[word] = freqlist[word] + 1 if word in freqlist else 1

if __name__ == '__main__':
    input_path  = '/share/nas165/amian/experiments/speech/espnet_old/egs2/aishell/asr1/data/aishell_data/train/text'
    output_path = './word_freq.txt'
    entity_path = './blists/aishell/test_1_entities.txt'

    jieba.load_userdict(entity_path)
    transcripts = read_file(input_path, sp=' ')
    transcripts = [" ".join(jieba.cut(text, cut_all=False)) for index, text in transcripts]

    freqlist = {}
    for text in transcripts:
        word_frequency(text, freqlist)

    freqlist = sorted([[word, str(freqlist[word])] for word in freqlist])

    time_now = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    exp_dir  = os.path.join(OUTPUT_DIR, time_now)
    os.mkdir(exp_dir)
    print(f'save to {exp_dir}...')

    output_path = os.path.join(exp_dir, 'word_freq.txt')
    write_file(output_path, freqlist)
    output_path = os.path.join(exp_dir, 'config.json')
    write_json(output_path, {'path': input_path})