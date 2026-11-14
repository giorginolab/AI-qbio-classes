"""Rebuild the three class decks, and their teaching-notes decks, from the source decks in slides/ and old/.

    uv run --no-project --with python-pptx python classes90/make_decks.py [SOURCE_ROOT]

SOURCE_ROOT (default: the current directory) is the folder that holds `slides/0 Class 1.pptx` and
`old/ai-qbio-unimi_2026_r1.pptx`. The decks are written next to this script. Slide numbers below are
those of the source decks.
"""
import copy
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


def sub(prs, old, new):
    """Replace `old` by `new` in the one run, on a slide or in its notes, that contains it.
    An empty `new` that would leave a paragraph empty removes the paragraph."""
    hits = 0
    for slide in prs.slides:
        frames = [sh.text_frame for sh in slide.shapes if sh.has_text_frame]
        if slide.has_notes_slide:
            frames.append(slide.notes_slide.notes_text_frame)
        for tf in frames:
            for p in tf.paragraphs:
                for r in p.runs:
                    if old in r.text:
                        r.text = r.text.replace(old, new); hits += 1
                if not new and p.runs and not ''.join(r.text for r in p.runs).strip():
                    p._p.getparent().remove(p._p)
    assert hits == 1, (old, hits)


def clone(prs, n, texts, notes):
    """Append a copy of text-only slide `n` (1-based) with the run texts of its shapes replaced by `texts`
    (one list of paragraph strings per text shape, in shape order) and its notes by `notes`."""
    src = prs.slides[n - 1]
    assert all(r.reltype.endswith(('slideLayout', 'notesSlide')) for r in src.part.rels.values())
    new = prs.slides.add_slide(src.slide_layout)
    for sh in list(new.shapes):
        sh._element.getparent().remove(sh._element)
    for el in src.shapes._spTree.iterchildren():
        if el.tag.endswith(('}sp', '}grpSp', '}cxnSp', '}pic')):
            new.shapes._spTree.append(copy.deepcopy(el))
    shapes = [sh for sh in new.shapes if sh.has_text_frame]
    assert len(shapes) == len(texts)
    for sh, paras in zip(shapes, texts):
        ps = sh.text_frame.paragraphs
        for p in ps[len(paras):]:
            p._p.getparent().remove(p._p)
        for i, text in enumerate(paras):
            if i >= len(ps):                           # extra paragraph: copy the last one
                ps[-1]._p.addnext(copy.deepcopy(ps[-1]._p)); ps = sh.text_frame.paragraphs
            runs = ps[i].runs
            runs[0].text = text
            for r in runs[1:]:
                r._r.getparent().remove(r._r)
    new.notes_slide.notes_text_frame.text = notes


def make(src, order, out, handson=None, unhide=(), edits=(), subs=(), clones=(), moves=()):
    """Keep the slides in `order` (1-based numbers in `src`), in that order; drop the rest.
    Slides made by `clones` are numbered after the last slide of `src`. `moves` are
    (slide, shape name, top, height) in EMU, with numbers in `src`; height None keeps it."""
    prs = Presentation(str(ROOT / src))
    for c in clones:
        clone(prs, *c)
    for n, name, top, height in moves:
        sh = next(sh for sh in prs.slides[n - 1].shapes if sh.name == name)
        sh.top = top
        if height:
            sh.height = height
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
    for old, new in subs:
        sub(prs, old, new)
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


# Teaching notes ("sidecar" decks): one per class, explaining the ideas behind each part of its notebook.
# Built from the per-practical decks in slides/; slide numbers below are those of the source decks.
N1 = 'slides/1 Fish toxicity QSAR · Teaching notes.pptx'
N2 = 'slides/2 Neural networks and enzymes · Teaching notes.pptx'
N3 = 'slides/3 Recurrent networks and protein targeting · Teaching notes.pptx'

# Class 1: t01's notes, reordered to class1_basics. Part 1 (4-10); a new slide 30 for Parts 2-7, which
# Class 1.pptx teaches; the playground at the end of Part 6 (20, 22, 23); Part 8 (11, 12, 14); Part 9
# (15, 17, 18, 19, 16, 21, 25, 24); Part 10 (26); homework and summary. Dropped: 13, t01's loss landscape
# (class 1 Part 4 draws its own, with other numbers).
make(N1, R(1, 10) + [30, 20, 22, 23, 11, 12, 14, 15, 17, 18, 19, 16, 21, 25, 24] + R(26, 29),
     'Class 1 · Teaching notes.pptx',
     # the longer title needs two lines: cover laid out as in class 3
     moves=[(1, 'Text 0', 2179320, None), (1, 'Text 1', 2717800, 1397000), (1, 'Text 2', 4343400, None)],
     clones=[(6, [['PARTS 2–7 · WITH THE LECTURE DECK'],
                  ['From a line to a network: the slides make a claim, the notebook measures it'],
                  ['Regression as a box (Part 2), one artificial neuron (Part 3), gradient descent on a loss '
                   'landscape (Part 4), a network and its width (Part 5), overfitting and the test set (Part 6) '
                   'and featurization (Part 7) are taught on slides 12–29 of Class 1.pptx.'],
                  ['These notes resume with the overfitting playground at the end of Part 6, then follow the '
                   'second half of the notebook (Parts 8–10): the same data treated as a practical problem.']],
              'Numbers the notebook prints in Parts 2–7: cross-validated R² 0.41 with MLOGP alone and 0.56 '
              'with all six descriptors (MLOGP coefficient +0.39). The identity neuron reproduces '
              'LinearRegression exactly; a fitted ReLU neuron gives the same line (w = 0.662, b = 2.668). '
              'Gradient descent diverges at lr = 0.2 because the valley is long and narrow (MLOGP is not '
              'centred). Width scan: 1 unit 0.557 (linear), 2 units −0.019 (dead ReLUs, an optimisation '
              'failure), 20 units 0.598, 100 units 0.571. Overfitting: test MSE turns around at epoch 218 '
              '(0.885 → 1.109); stopping there uses the test set to choose, so its score is biased: three '
              'splits. Featurization: random features and shuffled labels give R² ≈ 0.')],
     subs=[('PRACTICAL 1', 'HANDS ON/1'),
           ('Fish toxicity QSAR', 'Fish toxicity: from a line to a network, and how to trust it'),
           ('Repeat the analysis on a new species, then meet neural networks',
            'Repeat the analysis on a new species, then take networks further'),
           ('Companion to t01_toxicity_qsar.ipynb. These slides explain the general ideas behind each part '
            'of the notebook;',
            'Companion to classes90/class1_basics.ipynb. These slides explain the general ideas behind Part 1, '
            'the overfitting playground of Part 6 and Parts 8–10 of the notebook (Parts 2–7 follow the lecture '
            'deck, Class 1.pptx);'),
           ('Exercise 1.1: the counts', 'Warm-up: the counts'),
           ('(sessions 2 and 3)', '(classes 2 and 3)'),
           ('Neural networks (session 2) are a fourth family, fitted by gradient descent.',
            'Neural networks (Parts 3–5) are a fourth family, fitted by gradient descent.'),
           ('Exercise 5.1.', 'Exercise 6.'),
           ('Exercise 5.2:', 'Exercise 7:'),
           ('Exercise 5.3:', 'Exercise 4:'),
           ('Link to session 2: the alpha penalty', 'Link to class 2: the alpha penalty'),
           ('Homework (Part 7). ', 'Homework, at the end of the notebook. '),
           ('(session 2), other species (session 3)', '(class 2), other species (class 3)'),
           ('into the next sessions', 'into the next classes'),
           ('Session 2: a neural network on the same fish data, then enzymes from sequence.',
            'Class 2: a neural network on the same fish data, then enzymes from sequence.')])

# Class 2: t02's notes without overfitting epoch by epoch (12) and the MLP playground (13), which the
# class 2 notebook leaves out.
make(N2, R(1, 11) + R(14, 27), 'Class 2 · Teaching notes.pptx',
     subs=[('PRACTICAL 2', 'HANDS ON/2'),
           ('t02_neural_networks_enzymes.ipynb', 'classes90/class2_neural_networks_enzymes.ipynb'),
           ('data of session 1', 'data of class 1'),
           ('one-hot encoding of each position (session 3), embeddings from protein language models (session 4)',
            'one-hot encoding of each position (class 3), embeddings from protein language models (class 3 '
            'slides; the extra03 notebook)'),
           ('In the playground, compare a smaller network, more data and a penalty on the same toy curve.', ''),
           ('Train the same network with several seeds and average them: does the curve settle?', ''),
           ('Next session: a recurrent network', 'Next class: a recurrent network')])

# Class 3: t03's notes complete; the break comes after Part 5 (slide 19).
make(N3, R(1, 30), 'Class 3 · Teaching notes.pptx',
     subs=[('PRACTICAL 3', 'HANDS ON/3'),
           ('t03_rnn_protein_targeting.ipynb', 'classes90/class3_rnn_targeting.ipynb'),
           ('which is what protein language models produce (session 4)',
            'which is what protein language models produce (ProteinBERT, on the slides after the break)'),
           ('early stopping is still needed.',
            'early stopping is still needed. End of Part 5: the lecture slides on autoencoders and generative '
            'models, then the break.'),
           ('appeared in t02 with protein families held out', 'appeared in class 2 with protein families held out'),
           ('Session 4 uses ESM-2 embeddings.',
            'The lecture slides cover transformers and ProteinBERT; the extra03 notebook uses ESM-2 embeddings.'),
           ('In session 4, see how pre-trained protein language model embeddings change the picture.',
            'In the extra03 notebook, see how pre-trained protein language model embeddings (ESM-2) change '
            'the picture for the enzymes of class 2.')])
