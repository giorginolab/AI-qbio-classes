# Revision proposal for the three practicals

Status: sessions 1 and 2 are implemented as `s01_scaling_laws` and `s02_laws_vs_flexible_models`,
next to the old `t01`/`t02`, which are unchanged. The numbers in the notebooks and in `README.md`
supersede the preliminary ones below. Session 3 (trimming `t03`) and the transcription-factor
table are not done.

## The idea

Replace the toxicity material with **scaling laws across organisms**, and build the course around
one contrast: a *law* (two parameters, valid over many orders of magnitude) against a *flexible
model* (many parameters, valid only where it has seen data). Each session has one idea, one
dataset and one plot that students remember.

Toxicity is dropped entirely (t01, t02 part A, homework). Nothing in the new material involves
hazards, pathogens as a subject, or sequence design.

## Session 1: a law is a model

**Data:** AnAge (4645 animal species: body mass, metabolic rate, maximum longevity, taxonomy).

**Question on the board:** an elephant weighs 200 000 times as much as a mouse. How many times
more energy does it burn?

- Plot metabolic rate against body mass, first on linear axes (useless), then log-log (a line).
  Linear regression is introduced here, as the fit of an exponent.
- **Is it 2/3 or 3/4?** Mammals give 0.71 (n = 422, R² 0.95), birds 0.66. Surface-area geometry
  predicts 2/3, transport networks predict 3/4. Students bootstrap the slope and ask whether the
  data can tell the two theories apart. This is cross-validation and uncertainty with a real
  scientific dispute behind it.
- **The residuals are the biology.** Longevity scales with mass only loosely (mammals: slope 0.16,
  R² 0.50). The largest positive residuals are bats and humans. A model's errors are where the
  interesting questions start.
- **Pooling trap.** All classes together give R² 0.82 instead of 0.95: warm- and cold-blooded
  animals lie on parallel lines. One categorical variable repairs it.

## Session 2: what a flexible model cannot do

**Data:** the AnAge table again, plus NCBI genome reports (genome size and gene count for about
4700 eukaryotes and 13 800 complete prokaryote genomes; only those two columns and the group are
used).

**Question on the board:** train on animals lighter than 1 kg. Who predicts the whale?

- Fit a line, kNN, a random forest and a small neural network to metabolic rate. Inside the
  training range they are indistinguishable. Outside it, the forest and kNN go flat, the network
  does something arbitrary, and the two-parameter law is right. This one figure carries the
  session: interpolation is easy, extrapolation needs structure.
- **Species are not independent samples.** Random cross-validation puts sister species in training
  and test. Holding out whole orders gives the score for a genuinely new kind of animal. This is
  the same lesson as the family-grouped split in the current enzyme exercise, in a setting where
  everyone can see why.
- **A law that breaks.** In prokaryotes, gene number is proportional to genome size (slope 0.96,
  R² 0.98, about 940 genes per Mb). In eukaryotes the slope is 0.29 and the scatter is large.
  Same variables, different regime: students have to say what changed and why a model trained on
  one group fails on the other.
- *Optional, needs data preparation:* number of transcription factors against number of genes in
  bacteria grows roughly quadratically (van Nimwegen 2003), so regulation grows faster than the
  genome it regulates. I have not built or checked this table; it would come from UniProt
  keyword counts per reference proteome.

## Session 3: when there is no simple law

**Data:** the existing targeting-signal set, unchanged.

**Question on the board:** does the network read the letters, or their order?

Sequences have no two-parameter law, so here a neural network earns its place. Keep the core of
the current notebook and nothing else: composition baseline (0.80), GRU (0.94), GRU on shuffled
residues (0.80). The shuffled control is the point of the session. Hidden-state trajectories,
saliency and the playground become optional.

The enzyme-class exercise moves to homework, shortened to composition features with random
against family-grouped cross-validation.

## Format

- Each session opens with the question; students write a guess before any code runs.
- Solutions are collapsed, not shown directly under the exercise.
- Each session ends with one sentence the student writes: what the model can claim and what it
  cannot.

## Sources (all downloaded today without registration)

| Data | Source | Used for |
|---|---|---|
| AnAge build 15 | genomics.senescence.info/species/dataset.zip | sessions 1 and 2 |
| NCBI genome reports | ftp.ncbi.nlm.nih.gov/genomes/GENOME_REPORTS/ (`eukaryotes.txt`, `prokaryotes.txt`) | session 2 |
| PanTHERIA (mammal traits) | esapubs.org/archive/ecol/E090/184/ | spare: homework on litter size, home range, gestation |
| UniProt proteomes API | rest.uniprot.org/proteomes | optional transcription-factor table |

The prokaryote file is 217 MB and lists organism names, some of them pathogens. The course would
ship a reduced table with size, gene count and group only.

## Work involved

- New t01 and the first half of t02 are written from scratch; all numbers in the README for those
  parts are replaced. The fish and *Daphnia* files leave `data/`.
- t03 is trimmed, not rewritten; its numbers stay.
- A small `data_prep` script per new table, as for the existing datasets.
