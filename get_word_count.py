import io
import os
import pathlib

def get_transcript_text(transcript_path):
    with open(transcript_path) as f:
        return [line.strip().split(" ", 1)[1].lower() for line in f]

def get_transcripts(dataset_path):
    transcript_paths = dataset_path.glob("*/*/*.trans.txt")
    merged_transcripts = []
    for path in transcript_paths:
        merged_transcripts += get_transcript_text(path)
    return merged_transcripts

def read_file(path):
    datas = []
    with open(path, 'r', encoding='utf-8') as frs:
        for fr in frs:
            data = fr.replace('\n', '')
            data = data.split(' ')
            datas.append(data)
    return datas

def write_file(path, datas):
    with open(path, 'w', encoding='utf-8') as fr:
        for data in datas:
            fr.write(" ".join(data) + "\n")

def word_frequency(text, freqlist):
    for word in text.split(' '):
        freqlist[word] = freqlist[word] + 1 if word in freqlist else 1

if __name__ == '__main__':
    input_path  = '/share/corpus/LibriSpeech/LibriSpeech'
    output_path = './error_analysis/word_freq.txt'

    freqlist = {}
    for text in merged_transcripts:
        word_frequency(text, freqlist)

    freqlist = sorted([[word, str(freqlist[word])] for word in freqlist])
    write_file(output_path, freqlist)