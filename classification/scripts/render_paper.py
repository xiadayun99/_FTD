#!/usr/bin/env python3
"""Regenerate paper tables, a figure, and numerical checks from frozen summaries.

This script performs no GPU inference and creates no new experimental samples.
The full uploaded raw archives remain the source of measurement evidence.
"""
import collections
import csv
import hashlib
import json
import math
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

GIB = 2**30


def read(evidence, name):
    return json.loads((evidence / name).read_text())


def table(root, name, body):
    (root / 'tables' / name).write_text(body.strip() + '\n')


def render(root):
    evidence = root / 'evidence'
    (root / 'tables').mkdir(parents=True, exist_ok=True)
    (root / 'figures').mkdir(parents=True, exist_ok=True)
    fresh = read(evidence, 'fresh_confirmation_summary.json')
    freq = read(evidence, 'switch_frequency_audit.json')
    codec = read(evidence, 'cpu_zstd_summary.json')
    boots = read(evidence, 'fresh_stage_boots.json')
    assert fresh['status'] == 'FRESH_CONFIRMATION_COMPLETED'
    assert freq['status'] == 'LOCAL_AUDIT_PASSED' and freq['formal_stages'] == 54
    assert codec['status'] == 'LOSSLESS_BASELINE_COMPLETED'
    assert len(fresh['stages']) == 50 and fresh['formal_requests'] == 19200
    assert freq['formal_requests'] == 20736 and freq['distinct_questions'] == 384
    assert codec['configuration']['zstd_version'] == '1.5.5'
    for aggregate in fresh['aggregates'].values():
        assert aggregate['correct_each_block'] == [346] * 10
        assert aggregate['cache_hits_each_block'] == [378] * 10
    pretty = {'shared': 'Shared target', 'direct': 'Direct', 'node': 'Node',
              'native': 'HF native', 'host_copy': 'Host copy'}
    lines = [r'\begin{tabular}{lrrrrrrr}', r'\toprule',
             r'Method & Mean (ms) & P95 (ms) & Parameter (ms) & Score (ms) & Payload (GiB) & Fixed (GiB) & Allocated peak (GiB)\\',
             r'\midrule']
    for method in pretty:
        a = fresh['aggregates'][method]
        for key in ('mean_ms', 'mean_block_p95_ms', 'parameter_mean_ms', 'score_mean_ms'):
            raw = 'p95_ms' if key == 'mean_block_p95_ms' else key
            value = st.mean(x[raw] for x in fresh['stages'] if x['job']['method'] == method)
            assert math.isclose(value, a[key], rel_tol=1e-12, abs_tol=1e-10)
        values = [a[key] for key in ('mean_ms', 'mean_block_p95_ms', 'parameter_mean_ms', 'score_mean_ms')]
        values += [a[key] / GIB for key in ('retained_patch_bytes', 'fixed_tensor_bytes', 'allocator_peak_bytes')]
        lines.append(pretty[method] + ' & ' + ' & '.join(f'{x:.3f}' for x in values) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    table(root, 'confirmation50.tex', '\n'.join(lines))

    groups = [('canonical_mutable_raw_zstd', 'Canonical FP16'),
              ('canonical_mutable_xor_base_zstd', 'XOR to base'),
              ('direct_patch_zstd', 'Direct patches'),
              ('shared_patch_zstd', 'Shared patches')]
    lines = [r'\begin{tabular}{lrrr}', r'\toprule',
             r'Input & Raw (GiB) & Stored (GiB) & CPU restore (s)\\', r'\midrule']
    for key, name in groups:
        a = codec['groups'][key]
        assert a['raw_recovery_verified'] is True
        assert a['coded_bytes_plus_required_reference'] == a['coded_bytes_including_side_metadata'] + a['required_immutable_reference_file_bytes']
        lines.append(f'{name} & {a["raw_tensor_bytes"] / GIB:.3f} & {a["coded_bytes_plus_required_reference"] / GIB:.3f} & {a["cpu_restore_median_sum_ms"] / 1000:.3f}' + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    table(root, 'lossless_zstd.tex', '\n'.join(lines))

    regimes = ['low', 'medium', 'high']
    labels = ['Low (2)', 'Medium (47)', 'High (383)']
    lines = [r'\begin{tabular}{lrrr}', r'\toprule',
             r'Regime (switches) & Shared & Direct & HF native\\', r'\midrule']
    for regime, name in zip(regimes, labels):
        a = freq['aggregates'][regime]
        lines.append(name + ' & ' + ' & '.join(f'{a[m]["mean_ms"]:.3f}' for m in ('shared', 'direct', 'native')) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    table(root, 'frequency_mean.tex', '\n'.join(lines))

    sensitivity = []
    fresh_boot_order = list(dict.fromkeys(boots.values()))
    for boot in fresh_boot_order:
        blocks = [b for b in range(10) if all(boots[x['job']['tag']] == boot for x in fresh['stages'] if x['job']['block'] == b)]
        means = {m: st.mean(x['mean_ms'] for x in fresh['stages'] if x['job']['block'] in blocks and x['job']['method'] == m) for m in pretty}
        sensitivity.append(dict(study='Confirmation', boot_id=boot, blocks=blocks, mean_ms=means))
    for row in freq['complete_same_boot_block_sensitivity']:
        sensitivity.append(dict(study=row['regime'].title() + ' switch', boot_id=row['boot_id'],
                                blocks=row['complete_three_method_blocks'], mean_ms=row['mean_ms']))
    lines = [r'\begin{tabular}{llrrrrr}', r'\toprule',
             r'Study & Boot prefix & Complete blocks & Shared (ms) & Direct (ms) & HF native (ms) & Shared vs. Direct (\%)\\',
             r'\midrule']
    for row in sensitivity:
        a = row['mean_ms']
        delta = (a['shared'] / a['direct'] - 1) * 100
        lines.append(f'{row["study"]} & {row["boot_id"][:8]} & {len(row["blocks"])} & {a["shared"]:.3f} & {a["direct"]:.3f} & {a["native"]:.3f} & {delta:+.2f}' + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    table(root, 'reboot_sensitivity.tex', '\n'.join(lines))

    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 8,
                         'axes.labelsize': 8, 'axes.titlesize': 9, 'legend.fontsize': 8,
                         'xtick.labelsize': 7.6, 'ytick.labelsize': 7.6, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'axes.unicode_minus': False})
    fig, axes = plt.subplots(1, 2, figsize=(7.08, 2.65), sharey=True)
    positions = np.arange(3)
    width = .24
    colors = {'shared': '#2471A3', 'direct': '#D68910', 'native': '#616A6B'}
    hatches = {'shared': '', 'direct': '//', 'native': '..'}
    for ax, metric, title in zip(axes, ['mean_ms', 'p95_ms'], ['(a) Mean request latency', '(b) Mean stage p95 latency']):
        for index, method in enumerate(('shared', 'direct', 'native')):
            values, errors = [], []
            for regime in regimes:
                group = [x[metric] for x in freq['stages'] if x['job']['regime'] == regime and x['job']['method'] == method]
                assert len(group) == 6
                values.append(st.mean(group)); errors.append(st.stdev(group))
            ax.bar(positions + (index - 1) * width, values, width, yerr=errors,
                   color=colors[method], edgecolor='white', linewidth=.6, hatch=hatches[method],
                   error_kw={'ecolor': 'black', 'elinewidth': .7, 'capsize': 2}, label=pretty[method])
        ax.set_title(title, pad=7)
        ax.set_xticks(positions, labels)
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.grid(axis='y', color='#D8D8D8', linewidth=.45)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Latency (ms)')
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(.5, 1.01), handlelength=1.4, columnspacing=2.0)
    fig.supxlabel('Task switches per 384 requests', y=.015, fontsize=8)
    fig.subplots_adjust(left=.08, right=.985, bottom=.20, top=.78, wspace=.11)
    fig.savefig(root / 'figures/switch_frequency.pdf')
    fig.savefig(root / 'figures/switch_frequency.png', dpi=220)
    plt.close(fig)

    with (evidence / 'reported_measurements.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['study', 'regime', 'method', 'blocks', 'mean_ms', 'mean_stage_p95_ms',
                         'parameter_mean_ms', 'score_mean_ms', 'distinct_questions', 'correct_each_block'])
        for method, a in fresh['aggregates'].items():
            writer.writerow(['confirmation', '136 switches', method, 10, a['mean_ms'], a['mean_block_p95_ms'],
                             a['parameter_mean_ms'], a['score_mean_ms'], 384, 346])
        for regime in regimes:
            for method in ('shared', 'direct', 'native'):
                a = freq['aggregates'][regime][method]
                writer.writerow(['switch_frequency', regime, method, 6, a['mean_ms'], a['mean_block_p95_ms'],
                                 a['parameter_mean_ms'], a['score_mean_ms'], 384, 346])
    sd, ss = codec['groups']['direct_patch_zstd'], codec['groups']['shared_patch_zstd']
    checks = dict(status='PAPER_VALUES_REGENERATED_AND_CHECKED',
                  fresh_shared_vs_direct_percent=(fresh['aggregates']['shared']['mean_ms'] / fresh['aggregates']['direct']['mean_ms'] - 1) * 100,
                  fresh_shared_vs_node_latency_saving_percent=(1 - fresh['aggregates']['shared']['mean_ms'] / fresh['aggregates']['node']['mean_ms']) * 100,
                  fresh_shared_vs_node_extra_fixed_gib=(fresh['aggregates']['shared']['fixed_tensor_bytes'] - fresh['aggregates']['node']['fixed_tensor_bytes']) / GIB,
                  correction_payload_saving_percent=(1 - ss['raw_tensor_bytes'] / sd['raw_tensor_bytes']) * 100,
                  coded_shared_vs_coded_direct_saving_percent=(1 - ss['coded_bytes_plus_required_reference'] / sd['coded_bytes_plus_required_reference']) * 100,
                  fresh_boot_counts=dict(collections.Counter(boots.values())),
                  fresh_same_boot_blocks=sensitivity[:2],
                  frequency_boot_counts=freq['formal_stage_counts_by_boot'],
                  evidence_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence.glob('*.json') if p.name != 'generated_checks.json'})
    (evidence / 'generated_checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps({k: v for k, v in checks.items() if k not in ('evidence_sha256', 'fresh_same_boot_blocks')}, indent=2))
