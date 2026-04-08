# Experiment Report Guide

How to write experiment reports for the XTrans project.

## Template

Use `docs/report_template.html` as the starting point for every experiment report.
Copy it into the experiment directory as `analysis_report.html`.

```bash
cp docs/report_template.html exp_a_nccl_gates/analysis_report.html
```

## File Organization

Each experiment directory should contain:

```
exp_X/
├── README.md              # Experiment guide (goals, how to run)
├── analysis_report.html   # Final report (copy from template)
├── analysis_assets/       # Figures, diagrams (referenced by report)
│   ├── figure1.png
│   └── figure2.png
├── results/               # Raw data (gitignored)
│   ├── baseline.json
│   └── ...
└── [scripts/configs]      # Experiment-specific code
```

## Report Structure

Every report follows this section order (matching exp1 and exp2):

| # | Section | Required | Purpose |
|---|---------|----------|---------|
| 1 | **Objective** | Yes | What we're testing; one-paragraph hypothesis. Include a green highlight-box with the one-sentence result summary. |
| 2 | **Background** | If needed | Context the reader needs. Skip for follow-up experiments where prior reports cover this. |
| 3 | **Experimental Setup** | Yes | Hardware table + software table. Be exact: node name, GPU model, NCCL version, kernel version, container image tag. |
| 4 | **Configurations Under Test** | Yes | Table of all configs being compared. Each row: name, description, key settings, expected outcome. |
| 5 | **Results** | Yes | Data tables and figures. Use subsections (5.1, 5.2) for different result groups. Always show "vs Baseline" percentage. |
| 6 | **Analysis** | If needed | Deeper interpretation beyond what the numbers show. Explain mechanisms, surprises, limitations. Can merge into Results for simple experiments. |
| 7 | **Conclusions** | Yes | Numbered list of findings, each one sentence with key numbers. Include an info-box with next steps. |
| — | **Appendix A, B, ...** | If needed | Full data tables, config dumps, trace excerpts. Separated by `<hr class="appendix">`. |

## Style Rules

### Title and Subtitle

```html
<h1>Experiment A: NCCL Gate Taxonomy<br>and Minimal LD_PRELOAD Interception</h1>
<p class="subtitle">
  node192 &middot; 4&times; A100-SXM4-80GB &middot; NVLink NV12<br>
  Generated 2026-04-15 &middot; commit abc1234
</p>
```

- Title: `Experiment [letter]: [Descriptive Name]`
- Subtitle line 1: hardware summary
- Subtitle line 2: date and commit hash

### Callout Boxes

Three types, used sparingly:

```html
<!-- Green: key positive result or success -->
<div class="highlight-box">
  <p><b>Result:</b> The shim recovers 99.8% of NVLink bandwidth...</p>
</div>

<!-- Orange: warning, limitation, or risk -->
<div class="warning-box">
  <p><b>Limitation:</b> This was only tested on NCCL 2.21...</p>
</div>

<!-- Blue: informational note or next steps -->
<div class="info-box">
  <p><b>Next steps:</b> Test the DMA-BUF path on AMD hardware...</p>
</div>
```

Use in these locations:
- **highlight-box**: In Section 1 (result summary) and optionally in Section 5 (key result)
- **warning-box**: In Section 6 (limitations) or Section 5 (unexpected results)
- **info-box**: In Section 7 (next steps) or Section 2 (important context)

### Tables

Data tables: full-width, with header row. Use `font-variant-numeric: tabular-nums`
(already in the CSS) for aligned numbers.

```html
<table>
  <thead>
    <tr><th>Config</th><th>Bandwidth (GB/s)</th><th>vs Baseline</th></tr>
  </thead>
  <tbody>
    <tr><td><b>Baseline</b></td><td>156.6</td><td>100%</td></tr>
    <tr><td><b>Shim</b></td><td>155.9</td><td>99.6%</td></tr>
  </tbody>
</table>
```

Config/setup tables: use `class="config-table"` for key-value layouts.

```html
<table class="config-table">
  <tr><td>Node</td><td>node192</td></tr>
  <tr><td>GPU</td><td>4&times; A100-SXM4-80GB</td></tr>
</table>
```

### Figures

Store images in `analysis_assets/` next to the report. Reference with relative paths.

```html
<div class="figure">
  <img src="analysis_assets/bandwidth_chart.png" alt="Bandwidth comparison">
  <p>Figure 1: AllReduce bandwidth across configurations (1 KB – 1 GB).</p>
</div>
```

- Number figures sequentially: Figure 1, Figure 2, ...
- Captions should be self-contained (reader can understand without reading the text)
- Alt text should describe what the figure shows

### Code and Commands

Inline code: `<code>NCCL_CUMEM_ENABLE=1</code>`

Command blocks: use `<pre>` (no language highlighting needed):

```html
<pre>
LD_PRELOAD=./libxtrans_shim.so \
XTRANS_HOSTNAME=node192 \
  all_reduce_perf -b 1M -e 1G -g 1
</pre>
```

### Numbers and Units

- Bandwidth: GB/s (not Gbps)
- Latency: &mu;s or ms
- Message sizes: KB, MB, GB (powers of 1024)
- Percentages: one decimal place (99.5%, not 99.53218%)
- Use `&times;` for multiplication, `&ndash;` for ranges, `&ge;`/`&le;` for comparisons

## Naming Convention

- Report file: `analysis_report.html` (same name in every experiment directory)
- Assets directory: `analysis_assets/` (next to the report)
- Figure files: descriptive names like `bandwidth_chart.png`, `architecture_diagram.png`

## Checklist Before Finalizing

- [ ] Title matches experiment letter and descriptive name
- [ ] Subtitle has hardware, date, and commit hash
- [ ] Section 1 has a highlight-box with the one-sentence result
- [ ] Section 3 has exact hardware and software versions
- [ ] Section 4 has a configs table with all treatments
- [ ] Section 5 has "vs Baseline" column in every data table
- [ ] Section 7 conclusions are numbered, each one sentence
- [ ] All figures have numbered captions
- [ ] Appendix (if any) is separated by `<hr class="appendix">`
