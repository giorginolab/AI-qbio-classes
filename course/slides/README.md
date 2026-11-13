# Slide decks

Two decks per class, published as claude.ai Slides artifacts (private until shared from each deck's
Share menu; each can be presented in the browser or downloaded as PPTX or PDF).

## Practical decks

One idea per slide: the questions, exercises and figures of the notebook. Answers are in the
speaker notes or on the slide after the question.

| Class | Slides | Link |
|---|---|---|
| s01 Scaling laws | 21 | https://claude.ai/artifact/YR4QpeRQUazdDuCfvKixbh |
| s02 Laws vs flexible models | 25 | https://claude.ai/artifact/EAx5uHHwShFgC7thyZpmLh |
| s04 ESM-2 (optional) | 17 | https://claude.ai/artifact/2PghMjCiZQQy76Fwn8hKAZ |
| t01 Fish toxicity QSAR | 26 | https://claude.ai/artifact/KDPjKpBCnYpuDxyswV8d8E |
| t02 Neural networks and enzymes | 25 | https://claude.ai/artifact/GUPrqBKVWK9u2psBdrkXnN |
| t03 RNN protein targeting | 26 | https://claude.ai/artifact/BwFcYJUuEkahfJVfQ9X5KM |

## Teaching-notes decks

More text and theory: for each part, a concept slide with the general idea, the notebook figure as
evidence, and a "The general idea" takeaway.

| Class | Slides | Link |
|---|---|---|
| s01 Scaling laws | 27 | https://claude.ai/artifact/XUP7U7XiwMwaz1weXdT6Fs |
| s02 Laws against flexible models | 28 | https://claude.ai/artifact/GjrnJg6Fc2Ag8sJyPwjAEi |
| s04 A protein language model | 25 | https://claude.ai/artifact/WmQ5ub2ZrfDK9Mwa1RsZ6s |
| t01 Fish toxicity QSAR | 29 | https://claude.ai/artifact/1YuLnTJWQg7t9eZriS1FFZ |
| t02 Neural networks and enzymes | 27 | https://claude.ai/artifact/QgC4ZoFvFMhR8Guc8aPHjf |
| t03 Recurrent networks and protein targeting | 30 | https://claude.ai/artifact/JJRE7jLbS3LNR81d1g9aVD |

## Sources in this folder

```
practical/<class>/deck.json        title, slide order, sections, font
practical/<class>/slides/<id>.html one <section> per slide (1920 x 1080, inline styles, notes in <aside>)
practical/<class>/blobs.json       figure file -> asset URL it has in the published deck
teaching/<class>/...               same layout for the teaching-notes decks
assets/figures/                    notebook figures at 2x resolution (re-executed notebooks, RNG = 0)
assets/logo-*.{png,svg}            CNR and CNR-IBF logos from the CNR-IBF design system
figures/                           older figure exports, not used by these decks
```

The slide files are a snapshot of what was published (2026-11-11). Edits made in the browser are
not reflected here, and edits made here are not published automatically. In the slide files the
image paths point to `assets/`; to publish a file again, put back the `/_blob/...` URL from
`blobs.json` (and `project/ds/cnr-ibf/assets/` for the logos).

Numbers on the slides come from the notebooks and `course/README.md`. If seeds, splits or models
change, the slides need the same update as the notebook answers.
