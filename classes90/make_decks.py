"""Rebuild the three class decks from the two source decks (which are not in git).

    uv run --no-project --with python-pptx python classes90/make_decks.py [SOURCE_ROOT]

SOURCE_ROOT (default: the current directory) is the folder that holds `slides/0 Class 1.pptx` and
`old/ai-qbio-unimi_2026_r1.pptx`. The decks are written next to this script. Slide numbers below are
those of the source decks.
"""
import sys
from pathlib import Path
from pptx import Presentation

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
OUT = Path(__file__).parent
R = lambda a, b: list(range(a, b + 1))


def replace_text(prs, old, new):
    """Replace `old` by `new` in the one text run that contains it."""
    hits = 0
    for slide in prs.slides:
        for sh in slide.shapes:
            if sh.has_text_frame:
                for p in sh.text_frame.paragraphs:
                    for r in p.runs:
                        if old in r.text:
                            r.text = r.text.replace(old, new); hits += 1
    assert hits == 1, (old, hits)


def make(src, order, out, handson=None, unhide=(), edits=()):
    """Keep the slides in `order` (1-based numbers in `src`), in that order; drop the rest."""
    prs = Presentation(str(ROOT / src))
    lst = prs.slides._sldIdLst
    ids = list(lst)
    assert len(set(order)) == len(order)
    keep = [ids[i - 1] for i in order]
    for s in ids:
        if s not in keep:
            prs.part.drop_rel(s.rId)
        lst.remove(s)
    for s in keep:
        lst.append(s)
    for n in unhide:                                   # numbers in the source deck
        prs.slides[order.index(n)]._element.attrib.pop('show', None)
    if handson:
        replace_text(prs, 'Hands-on 1', handson)
    for old, new in edits:
        replace_text(prs, old, new)
    prs.save(str(OUT / out))
    print(out, len(order), 'slides')


# Class 1 (150'): the author's deck, with optimization and backpropagation moved before overfitting,
# and "Interpretability" (29) moved after featurization, where the notebook asks what the model learned.
# Text corrections: backpropagation lowers the error on the training set, not the test set.
make('slides/0 Class 1.pptx', R(1, 23) + [27, 28, 24, 25, 26] + [30, 31, 29, 32], 'Class 1.pptx',
     edits=[('decreases the error on the test set', 'decreases the error on the training set'),
            ('loss funtion', 'loss function')])

FULL = 'old/ai-qbio-unimi_2026_r1.pptx'
# Class 2 (90'): deephERG, deep networks, featurization, evaluation of classifiers (shown), Merck, CNNs.
make(FULL, [1, 60, 61, 33, 34, 30, 31] + R(81, 84) + R(35, 40), 'Class 2.pptx', 'Hands-on 2', unhide=[31])
# Class 3 (150'): RNNs, autoencoders and generative models, transformers, applications, final words.
make(FULL, [1] + R(41, 59) + R(62, 77) + [79, 80, 89, 90, 91, 78], 'Class 3.pptx', 'Hands-on 3',
     edits=[('performace', 'performance')])
