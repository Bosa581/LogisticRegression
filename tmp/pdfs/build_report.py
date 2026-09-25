import sys
import pathlib
import shutil

sys.path.insert(0, str(pathlib.Path('tmp/pdf_dependencies').resolve()))
import pymupdf

root = pathlib.Path('.')
log_text = (root / 'tmp/tfidf_run.txt').read_text(encoding='utf-16')
last_line = log_text.strip().splitlines()[-1]
fields = last_line.split()
assert fields[0] == 'Update'
assert fields[1] == '10640'
tfidf_loss = -float(fields[5]) / 133
tfidf_train = float(fields[7]) * 100
tfidf_accuracy = float(fields[9]) * 100

backup = root / 'tmp/pdfs/report_before_tfidf.pdf'
if not backup.exists():
    shutil.copyfile(root / 'report.pdf', backup)

doc = pymupdf.open()
page = doc.new_page(width=612, height=792)
y = 42


def paragraph(text, size=10.5, font='tiro', gap=7):
    global y
    box = pymupdf.Rect(48, y, 564, 752)
    remaining = page.insert_textbox(box, text, fontsize=size, fontname=font,
                                    lineheight=1.15)
    if remaining < 0:
        raise ValueError('Report does not fit on one page')
    y = 752 - remaining + gap


paragraph('Logistic Regression: Baseball or Hockey', size=14, font='tibo', gap=3)
paragraph('Orobosa Igbinovia | COMP5420 | Assignment 3', size=10, gap=10)
paragraph('I used logistic regression with stochastic gradient updates to predict baseball (1) or hockey (0). '
          'All experiments used seed 1701, the same 1,064 training documents and 133 held-out documents, '
          'and ten passes. The baseline features were word counts plus a bias. The learning rate controls '
          'the size of each weight update: smaller rates learn more slowly, while larger rates can fluctuate.')

paragraph('After ten passes, word-count models with learning rates 0.01, 0.1 and 0.5 reached '
          '94.74%, 94.74% and 93.23% held-out accuracy, with mean log losses of 0.185, 0.145 and 0.238. '
          'Lazy L2 (step 0.1, mu = 0.001) reached 93.98% accuracy and 0.173 loss; the decreasing rate '
          'reached 94.74% and 0.297. TF-IDF at step 0.1 reached '
          f'{tfidf_accuracy:.2f}% accuracy and {tfidf_loss:.3f} loss. All losses exclude the L2 penalty.')

paragraph('After one pass, rates '
          '0.01, 0.1 and 0.5 gave 91.73%, 93.98% and 93.23% held-out accuracy. Rate 0.1 had the lowest '
          'final loss. Its training accuracy reached 99.91% at pass four and stayed there. Held-out '
          'accuracy was 94.74% from pass two onward except for 95.49% at pass five. About four passes '
          'were enough for accuracy to settle, although weights and loss continued changing.')
paragraph('For extra credit, the decreasing rate was 0.1 / sqrt(t), where t is the update number starting '
          'at 1; training accuracy reached 95.58%. Lazy L2 delayed decay for inactive features and applied '
          'remaining decay before evaluation. It reached 96.24% held-out accuracy in passes three through '
          'seven, but the gain did not last. It still scans all features, so I did not establish a speed advantage.')
paragraph('For TF-IDF, I used tf = 1 + ln(count) for present words and idf = 1 + ln((N + 1) / (df + 1)), '
          'with N = 1,197 and df from the supplied vocabulary. Absent words have value zero. I multiplied '
          'tf by idf, divided the word vector by its L2 length, and kept the bias at 1. Training and evaluation '
          'both use this vector. At step 0.1, TF-IDF reached '
          f'{tfidf_train:.2f}% training and {tfidf_accuracy:.2f}% held-out accuracy. Its loss of {tfidf_loss:.3f} '
          'was higher than the count baseline: better classification accuracy did not mean better probability '
          'predictions. Normalization also changes feature scale, so the same step size need not be optimal.')
paragraph('To explain the baseline predictions, I ranked coefficients after ten passes at step 0.1, excluding '
          'the bias. The strongest baseball words were "runs" (+1.815), "pitching" (+1.523) and "baseball" '
          '(+1.217). The strongest hockey words were "hockey" (-2.925), "playoffs" (-1.996) and "golchowy" '
          '(-1.547). Mathematically, log(p / (1 - p)) = b + sum(w_j x_j): each extra word occurrence changes '
          'baseball log-odds by its coefficient, holding other counts fixed. Ranking training-observed words '
          'by absolute coefficient identifies weak predictors; "dive," "privately" and "kills" had weights '
          'approximately zero.')
paragraph('These features ignore word order and context. All comparisons reuse one small held-out split; '
          'one document changes accuracy by about 0.75 percentage points. TF-IDF uses the supplied corpus '
          'frequencies, which include held-out documents; a strictly isolated evaluation would fit IDF on '
          'training documents only. The accuracy differences therefore deserve caution.', gap=0)

doc.set_metadata({'title': 'Assignment 3: Logistic Regression', 'author': 'Orobosa Igbinovia'})
doc.save(root / 'report.pdf')
page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(root / 'tmp/pdfs/report_updated.png')
print('Report pages:', len(doc), 'Content bottom:', y)
print('TF-IDF:', tfidf_train, tfidf_accuracy, tfidf_loss)
