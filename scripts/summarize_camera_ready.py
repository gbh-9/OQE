"""Generate the joint reviewer table and report from the fresh matched timing cohort."""
import csv
import json
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-dir', type=Path, default=ROOT/'reference/camera_ready_20261029')
parser.add_argument('--output-dir', type=Path, default=ROOT/'results/camera_ready_tables')
args = parser.parse_args()
RUN = args.run_dir
OUT = args.output_dir
OUT.mkdir(parents=True, exist_ok=False)
with (RUN/'timing_summary.csv').open(newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
with (ROOT/'reference/camera_ready_20261029/representation_summary.csv').open(newline='', encoding='utf-8') as f:
    representation = list(csv.DictReader(f))

def get(scope, device, batch, method, field='median_ms'):
    found = [r for r in rows if (r['scope'],r['device'],int(r['batch']),r['method']) == (scope,device,batch,method)]
    assert len(found)==1
    return float(found[0][field])

def pair(scope, device, method, field='median_ms', scale=1, digits=3):
    return ' / '.join(f'{get(scope,device,b,method,field)/scale:.{digits}f}' for b in (1,128))

lines = [r'\begin{table*}[t]',r'\centering',
    r'\caption{Joint representation and execution trade-offs: $K=7$, two anchors, Fashion-MNIST.}',
    r'\label{tab:deployment}',r'\scriptsize',r'\setlength{\tabcolsep}{5pt}',
    r'\begin{tabular}{lrrrr}',r'\toprule',
    r'Encoding & Clean accuracy (\%) & Tail events/scalar & int64 keys, $B=128$ & Packed bound, $B=128$ \\',r'\midrule']
for enc, label in [('oqe','OQE'),('single_hybrid','Matched Single-Spike')]:
    r=next(r for r in representation if r['encoding']==enc)
    lines.append(f"{label} & ${float(r['ordinary_clean_accuracy_percent']):.2f}\\pm{float(r['ordinary_clean_population_std_pp']):.2f}$ & {float(r['tail_events_per_scalar']):.3f} & {int(r['int64_input_key_bytes_B128'])/1e6:.3f} MB & {int(r['packed_3bit_key_bound_bytes_B128'])/1e6:.4f} MB "+r'\\')
lines += [r'\bottomrule',r'\end{tabular}',r'\par\smallskip',r'\setlength{\tabcolsep}{4pt}',
    r'\begin{tabular}{lrrrrr}',r'\toprule',
    r'& \multicolumn{2}{c}{Reconstruction (ms)} & \multicolumn{2}{c}{Full forward (ms)} & GPU extra peak \\',
    r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
    r'Execution path & CPU & GPU & CPU & GPU & allocation (MB) \\',r'\midrule']
names=[('oqe_ordinary','OQE ordinary'),('oqe_generic','OQE generic grouped'),('oqe_suffix','OQE suffix grouped'),
       ('single_ordinary','Single ordinary'),('single_grouped','Single grouped')]
for method,label in names:
    recon=['--','--'] if method.endswith('ordinary') else [pair('reconstruction',dev,method) for dev in ('cpu','cuda:0')]
    lines.append(' & '.join([label,*recon,pair('full','cpu',method),pair('full','cuda:0',method),
        pair('full','cuda:0',method,'incremental_peak_bytes',1e6,2)])+r' \\')
lines += [r'\bottomrule',r'\end{tabular}',r'\par\smallskip',r'\begin{minipage}{0.98\textwidth}\scriptsize',
    r'Each pair is batch 1 / 128. Accuracy: ordinary-forward three-seed mean $\pm$ population standard deviation;',
    r'events: six-step tail, excluding anchors. Timings: same-session medians, CPU 5/GPU 20 repetitions;',
    r'full paths use each encoding\textquotesingle s frozen seed-20261010 model and identical images.',
    r'Reconstruction uses synthetic preformed aggregates; -- means no separate grouped stage.',
    r'GPU extra peak: separately measured peak allocated bytes above the post-reset live baseline, not total device memory.',
    r'Keys: input-class tensor only; packing is a theoretical bound, not implemented ($10^6$ bytes/MB).',
    r'\end{minipage}',r'\end{table*}','']
(OUT/'deployment.tex').write_text('\n'.join(lines),encoding='utf-8')
report=['# Frozen-checkpoint deployment benchmark (2026-10-29)','',
    'No training; all 5 paths measured together. CPU warmup 1/repeats 5; GPU warmup 5/repeats 20. FP32, eval/no_grad, reset inside timed call, synchronization at timing boundaries. Input and checkpoints are resident; data loading/H2D excluded. Timing repetitions are not independent training seeds. Desktop background load is not isolated; raw samples and dispersion are retained.','',
    '| Device | Batch | OQE ordinary | OQE generic | OQE suffix | Single ordinary | Single grouped | Suffix/ordinary OQE |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for dev in ('cpu','cuda:0'):
    for b in (1,128):
        values=[get('full',dev,b,m) for m,_ in names]
        report.append('| '+' | '.join([dev,str(b),*[f'{v:.3f}' for v in values],f'{values[2]/values[0]:.3f}'])+' |')
report += ['', '## Reconstruction medians (ms)', '', '| Device | Batch | OQE generic | OQE suffix | Single placement |', '|---|---:|---:|---:|---:|']
for dev in ('cpu','cuda:0'):
    for b in (1,128):
        report.append('| '+' | '.join([dev,str(b),*[f"{get('reconstruction',dev,b,m):.3f}" for m in ('oqe_generic','oqe_suffix','single_grouped')]])+' |')
report += ['', '## Memory and interpretation', '', '| Path | GPU extra peak MB, B1 | GPU extra peak MB, B128 |','|---|---:|---:|']
for m,label in names:
    report.append('| '+label+' | '+' | '.join(f"{get('full','cuda:0',b,m,'incremental_peak_bytes')/1e6:.3f}" for b in (1,128))+' |')
report += ['', '1. Suffix reconstruction is faster than generic reconstruction in all 4 measured settings. This is a stage-level result.',
    '2. Complete grouped forwarding remains slower than ordinary forwarding. Dense ordinary paths do not exploit the event-count difference as sparse kernels would; lower event activity alone does not establish proportional latency savings.',
    '3. Extra peak allocation includes all live transient tensors/output above a reset baseline, not just metadata. Similar peaks across encodings are consistent with shared dense tensor shapes. No hardware energy claim follows.',
    '4. All 12 matched-path batch checks had zero prediction and final-rate differences; these checks cover first B=1/128 batches only, not a new full-test accuracy evaluation.',
    '', '## Instrumentation changelog', '',
    '| File | Change |', '|---|---|',
    '| scripts/benchmark_oqe_camera_ready.py | New standalone wrapper; frozen load, randomized timing, separate memory pass, local Single grouped path, source/checkpoint hashes. Existing model/executor files unchanged. |',
    '| scripts/summarize_camera_ready.py | Portable release adaptation of the CSV-driven camera-ready table/report generator. |',
    '| deployment.tex in the requested output directory | Generated deployment table. |',
    '', 'The preliminary run `oqe_camera_ready_20261029` is excluded: its Single wrapper retained grouping temporaries during downstream execution. The final helper releases them at the same boundary as OQE; all paths were then remeasured together in this final cohort. Historical v23 timings remain preserved and are not mixed into the new table.',
    '', 'This report regenerates the timing summaries without training. The clean-accuracy/event/key-size block comes from the frozen reference representation summary, not the timing run.','']
(OUT/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')
print('\n'.join(report[:11]))
print('SINGLE_RECON_BETTER',sum(get('reconstruction',d,b,'single_grouped')<get('reconstruction',d,b,'oqe_suffix') for d in ('cpu','cuda:0') for b in (1,128)))
