import sys
from pptx import Presentation
def make(src, order, out, handson=None):
    prs = Presentation(src)
    lst = prs.slides._sldIdLst
    ids = list(lst)
    keep = [ids[i - 1] for i in order]
    assert len(set(order)) == len(order)
    for s in ids:
        if s not in keep: prs.part.drop_rel(s.rId)
        lst.remove(s)
    for s in keep: lst.append(s)
    if handson:
        n = 0
        for sh in prs.slides[0].shapes:
            if sh.has_text_frame:
                for p in sh.text_frame.paragraphs:
                    for r in p.runs:
                        if 'Hands-on 1' in r.text: r.text = r.text.replace('Hands-on 1', handson); n += 1
        assert n == 1, n
    prs.save(out); print(out, len(order), 'slides')
R = lambda a, b: list(range(a, b + 1))
make('slides/0 Class 1.pptx', R(1, 23) + [27, 28, 24, 25, 26] + R(29, 32), 'classes90/Class 1.pptx')
full = 'old/ai-qbio-unimi_2026_r1.pptx'
make(full, [1, 30, 32] + R(33, 52), 'classes90/Class 2.pptx', 'Hands-on 2')
make(full, [1] + R(53, 70) + R(81, 84) + [87] + R(71, 77) + [79, 80, 89, 90, 91, 78], 'classes90/Class 3.pptx', 'Hands-on 3')
