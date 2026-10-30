"""Frozen-checkpoint, same-session deployment comparison; no training or source mutation."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import platform
import random
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import torch
import torchvision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from oqe_v6 import MatchedInputSNN, codebook_for, reset_model
from oqe_v6_executor import (classwise_tail_aggregates, generic_reconstruct_tail_currents,
    suffix_reconstruct_tail_currents, single_spike_reconstruct_tail_currents,
    quotient_forward, suffix_quotient_forward)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def single_first_currents(model, images):
    ids = (images.unsqueeze(0) >= model.encoder.thresholds.reshape(-1, 1, 1, 1, 1)).sum(0).long()
    anchor = model.conv1(images.unsqueeze(0))
    aggregates = classwise_tail_aggregates(ids, model.conv1.weight, k=model.k, padding=1)
    tail = single_spike_reconstruct_tail_currents(aggregates)
    return torch.cat((anchor.expand(2, -1, -1, -1, -1), tail), 0)


def single_grouped(model, images):
    # Match the OQE executor's function boundary: release grouping temporaries
    # before downstream inference, rather than retaining them in this frame.
    return model.forward_from_first_currents(single_first_currents(model, images))


def load(encoding, device):
    path = ROOT / 'results/oqe_v6_main/checkpoints' / f'fashion_mnist__{encoding}__k7__canonical__s20261010__e3.pt'
    payload = torch.load(path, map_location='cpu', weights_only=False)
    assert payload['k'] == 7
    model = MatchedInputSNN(encoding=encoding, input_channels=1, output_classes=10,
        image_size=28, thresholds=payload['thresholds'], k=7)
    model.load_state_dict(payload['model_state'], strict=True)
    assert torch.equal(model.encoder.codebook, codebook_for(encoding, 7))
    return model.to(device).eval(), path


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interleaved(functions, device, seed):
    warmup, repeats = (5, 20) if device.type == 'cuda' else (1, 5)
    for fn in functions.values():
        for _ in range(warmup):
            fn()
    sync(device)
    samples = {name: [] for name in functions}
    rng = random.Random(seed)
    for _ in range(repeats):
        order = list(functions)
        rng.shuffle(order)
        for name in order:
            sync(device)
            t0 = time.perf_counter()
            functions[name]()
            sync(device)
            samples[name].append((time.perf_counter() - t0) * 1000)
    return samples, warmup, repeats


@torch.no_grad()
def run(output):
    output.mkdir(parents=True, exist_ok=False)
    metadata = {'python': sys.version, 'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'platform': platform.platform(), 'cpu_threads': torch.get_num_threads(),
        'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_tf32': torch.backends.cudnn.allow_tf32,
        'precision': 'FP32; no autocast', 'training': False, 'dataset': 'Fashion-MNIST test first B examples',
        'seed': 20261010, 'K': 7, 'anchors': 2,
        'timing_scope': 'reset + resident-input forward; excludes data loading, H2D and checkpoint loading',
        'schedule': 'fixed-seed randomized method order per repetition; separate warmups per method',
        'memory_scope': 'separate untimed CUDA pass: peak allocated minus post-reset live baseline; not nvidia-smi or reserved memory',
        'sources': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT/'src/oqe_v6.py', ROOT/'src/oqe_v6_executor.py']}}
    metadata['gpu_snapshot_before'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.used,utilization.gpu', '--format=csv,noheader'], text=True).strip()
    dataset = torchvision.datasets.FashionMNIST(ROOT/'data/fashion_mnist', train=False,
        transform=torchvision.transforms.ToTensor(), download=False)
    inputs = torch.stack([dataset[i][0] for i in range(128)])
    metadata['input_sha256'] = hashlib.sha256(inputs.numpy().tobytes()).hexdigest()
    rows, raw, checks = [], [], []
    for device in [torch.device('cpu'), torch.device('cuda:0')]:
        assert device.type != 'cuda' or torch.cuda.is_available()
        oqe, oqepath = load('oqe', device)
        single, singlepath = load('single_hybrid', device)
        assert torch.equal(oqe.encoder.thresholds, single.encoder.thresholds)
        metadata['sources'].update({str(p.relative_to(ROOT)): sha(p) for p in [oqepath, singlepath]})
        for batch in [1, 128]:
            images = inputs[:batch].to(device)
            def wrapped(model, function):
                def call():
                    reset_model(model)
                    return function(model, images)
                return call
            functions = {
                'oqe_ordinary': wrapped(oqe, lambda m, x: m(x)),
                'oqe_generic': wrapped(oqe, quotient_forward),
                'oqe_suffix': wrapped(oqe, suffix_quotient_forward),
                'single_ordinary': wrapped(single, lambda m, x: m(x)),
                'single_grouped': wrapped(single, single_grouped),
            }
            outputs = {name: fn().cpu() for name, fn in functions.items()}
            for name, reference in [('oqe_generic','oqe_ordinary'), ('oqe_suffix','oqe_ordinary'), ('single_grouped','single_ordinary')]:
                mismatch = int((outputs[name].argmax(1) != outputs[reference].argmax(1)).sum())
                checks.append({'device': str(device), 'batch': batch, 'method': name,
                    'prediction_mismatches': mismatch, 'rate_max_abs': float((outputs[name]-outputs[reference]).abs().max())})
                assert mismatch == 0, checks[-1]
            del outputs
            samples, warmup, repeats = interleaved(functions, device, 291000+batch)
            for name, values in samples.items():
                baseline, peak, extra = '', '', ''
                if device.type == 'cuda':
                    reset_model(oqe)
                    reset_model(single)
                    gc.collect()
                    sync(device)
                    baseline = torch.cuda.memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    result = functions[name]()
                    sync(device)
                    peak = torch.cuda.max_memory_allocated(device)
                    extra = peak - baseline
                    del result
                rows.append({'scope': 'full', 'device': str(device), 'batch': batch, 'method': name,
                    'median_ms': st.median(values), 'mean_ms': st.mean(values), 'std_ms': st.pstdev(values),
                    'warmup': warmup, 'repeats': repeats, 'baseline_allocated_bytes': baseline,
                    'peak_allocated_bytes': peak, 'incremental_peak_bytes': extra})
                raw.extend({'scope':'full','device':str(device),'batch':batch,'method':name,'repeat':i,'ms':v} for i,v in enumerate(values))
            reset_model(oqe)
            reset_model(single)
            generator = torch.Generator().manual_seed(17007+batch)
            aggregates = torch.randn(7, batch, 16, 28, 28, generator=generator).to(device)
            aggregates[0].zero_()
            book = codebook_for('oqe',7,device=device,dtype=aggregates.dtype)
            micros = {'oqe_generic': lambda: generic_reconstruct_tail_currents(aggregates,book),
                'oqe_suffix': lambda: suffix_reconstruct_tail_currents(aggregates),
                'single_grouped': lambda: single_spike_reconstruct_tail_currents(aggregates)}
            samples, warmup, repeats = interleaved(micros,device,291100+batch)
            for name, values in samples.items():
                rows.append({'scope':'reconstruction','device':str(device),'batch':batch,'method':name,
                    'median_ms':st.median(values),'mean_ms':st.mean(values),'std_ms':st.pstdev(values),
                    'warmup':warmup,'repeats':repeats,'baseline_allocated_bytes':'','peak_allocated_bytes':'','incremental_peak_bytes':''})
                raw.extend({'scope':'reconstruction','device':str(device),'batch':batch,'method':name,'repeat':i,'ms':v} for i,v in enumerate(values))
            del functions, micros, aggregates, book, images
            write_csv(output/'timing_summary.csv',rows)
            write_csv(output/'timing_samples.csv',raw)
            print(f'completed {device} B={batch}',flush=True)
        del oqe,single
        gc.collect()
        if device.type=='cuda':
            torch.cuda.empty_cache()
    metadata['gpu_snapshot_after'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.used,utilization.gpu', '--format=csv,noheader'], text=True).strip()
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    (output/'equivalence_checks.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'COMPLETE','rows':len(rows),'checks':len(checks),'output':str(output)}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.output.resolve())
