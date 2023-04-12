from Bio    import pairwise2

# target = "不可以給我們今年的那個htr因為去年家樂福那個是htr"
# texts   = ['可能會給我們今年的那個htr因為去年家樂福那個是htr', '可不可以給我們今年的那個htr因為去年家樂福那個是wealthmanagement', '可不可以給我們今年的那個htrok因為去年家樂福那個是htr', '可不可以給我們今年的那個htrok因為去年家樂福那個是wealthmanagement', '可不可以給我們今年的那個整個的因為去年家樂福那個是wealthmanagement', '可不可以 給我們今年的那個htr因為去年家樂福那個是wealthmanageratio', '可不可以給我們今年的那個整個ok的因為去年家樂福那個是wealthmanagement', '可不可以給我們今年的那個整個的因為去年家樂福那個是wealthmanageratio', '可不可以給我們今年的那個htrok因為去年家樂福那個是wealthmanageratio']

target = "我是美林的cathrine"
text  = "我是美林的"

alignments = pairwise2.align.localms(target, text,2,-1,-0.5,-0.1, one_alignment_only=True)
print(alignments)
# alignments = [pairwise2.align.globalxx(target, text)[0] for text in texts]

# alignments = [[align.seqA, align.seqB] for align in alignments]

# for seqA, seqB in alignments:
#     print(seqA)
#     print(seqB)
#     print('_' * 30)

from Bio import Align

s = '我是美林的cathrine'
t = '我是美林的'

def get_aligner():
    aligner = Align.PairwiseAligner()

    aligner.match_score = 1.0 
    aligner.mismatch_score = -2.0
    aligner.gap_score = -2.5
    return aligner

aligner = get_aligner()
alignments = aligner.align(s, t)



# for all the alignments
# print(*alignments, sep='\n')

for align in sorted(alignments):
    print(str(align))
