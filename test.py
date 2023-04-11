from Bio import pairwise2

def text_normalize(text):
    return text.replace('-', ' ')

def aligment(target, text):
    alignments = pairwise2.align.globalxx(a, b)[0]
    return text_normalize(alignments.seqA), text_normalize(alignments.seqB)

a = "AXI比較特別的是它可以用X ray的原理"
b = "AXI比較特別的是以用X_y ray的原理"

a, b = aligment(a, b)
print(a)
print(b)