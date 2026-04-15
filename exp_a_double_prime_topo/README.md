# Exp A'': NCCL Topology Virtualization

**Phase 1.75 of the XTrans research plan (Section 6.2.1)**

## Goal

Validate that a daemon-generated NCCL topology XML file can make per-GPU containers
see the full NVLink topology and select the correct P2P transport, without relying
on host `/sys` visibility.

## Why This Matters

Exp A proved that syscall interception (hostHash, shmDev) enables cross-container IPC.
But NCCL also reads `/sys/bus/pci` to discover GPU topology (NVLink connections, PCIe
hierarchy, NUMA affinity). In per-GPU containers, this sysfs view is incomplete — each
container sees only its own GPU's PCI subtree.

This experiment validates the *first layer* of the XTrans architecture (L1: Topology
Service) on NVIDIA. If `NCCL_TOPO_FILE` works reliably as the topology injection
mechanism, the daemon's topology service reduces to "enumerate GPUs via NVML → generate
XML → serve to containers." If it *doesn't* work (e.g., NCCL cross-validates XML against
sysfs), we discover this now rather than during Phase 3 daemon integration.

Key background:
- NCCL's topology loading chain: (1) `NCCL_TOPO_FILE` env var, (2) fallback to
  `/var/run/nvidia-topologyd/virtualTopology.xml`, (3) auto-detect from `/sys`
- `nvidia-topologyd` is NVIDIA's undocumented daemon that does exactly this for VMs
- Cloud providers (Nebius, Azure, Crusoe) already ship per-SKU topology XMLs
- NCCL Issue #326 documents crashes when container `/sys` is incomplete

## Success Criteria

- Daemon-generated `NCCL_TOPO_FILE` + Exp A shim recovers ≥99% of bare-metal
  NVLink allreduce bandwidth in per-GPU containers on node192
- NCCL selects the identical transport graph (P2P/NVLink rings) as bare metal

## Hardware

- node192: 4x NVIDIA A100 80GB, NVLink 3.0

## Tasks

### A''1: NCCL Topology XML Reverse-Engineering

Dump NCCL's auto-detected topology XML on node192 (bare metal, 4×A100 NVLink):

```bash
# Dump NCCL's auto-detected topology
NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo.xml \
  NCCL_DEBUG=INFO \
  all_reduce_perf -b 1M -e 1G -g 4
```

Document the XML schema:
- `<system>`, `<cpu>`, `<pci>`, `<gpu>`, `<nvlink>` elements
- Which fields are required vs optional for transport selection
- Compare XML output with `/sys/bus/pci` sysfs tree

**Deliverable**: Complete NCCL topology XML schema documentation. Reference XML
for node192 A100 topology.

### A''2: Daemon-Generated Topology XML

Write `topo_gen_nvidia.py` that uses NVML (`pynvml`) to enumerate:
- GPUs (count, PCI bus IDs, compute capability)
- NVLink connections (link count, bandwidth per link, peer GPU)
- PCIe topology (root complex, switches, slot positions)
- NUMA affinity

Generate an NCCL-compatible topology XML file. Validate by comparing NCCL's
transport decisions when using this file vs auto-detection.

**Deliverable**: `topo_gen_nvidia.py` — host-side topology XML generator,
validated against NCCL auto-detection on node192.

### A''3: Cross-Container Topology Injection

Run per-GPU containers (1 GPU each) with the shim from Exp A. Set `NCCL_TOPO_FILE`
pointing to the daemon-generated XML (bind-mounted into each container).

Configurations to compare:
1. **Bare metal** (baseline): all GPUs visible, no containers
2. **Shim only**: Exp A shim, no topology file
3. **Shim + topo file**: Exp A shim + `NCCL_TOPO_FILE`
4. **`--ipc=host`**: destroy isolation entirely (upper bound)

Verify transport selection via `NCCL_DEBUG=INFO`:
- P2P/NVLink selected (not NET/Socket)?
- Correct ring/tree algorithm?

Measure allreduce bandwidth (1MB–1GB) for each configuration.

**Deliverable**: Proof that daemon-generated topology XML + shim enables correct
transport selection in per-GPU containers. Bandwidth comparison table.

### A''4: Topology Correctness Validation

Compare NCCL's transport graph across configurations:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH \
  NCCL_GRAPH_DUMP_FILE=/tmp/nccl_graph.txt \
  NCCL_TOPO_FILE=/path/to/generated.xml \
  all_reduce_perf -b 1M -e 1G -g 4
```

Verify:
- NVLink ring/tree topology matches bare metal
- Same number of channels and rings
- Same algorithm selection (ring vs tree vs split tree)

**Deliverable**: Transport graph equivalence proof. Document any topology fields
that affect algorithm selection.

### A''5: Incomplete/Incorrect Topology Resilience

Test NCCL behavior with deliberately flawed topology XMLs:
1. Missing NVLink entries (claim fewer links than exist)
2. Incorrect bandwidth values (10x too high, 10x too low)
3. Extra phantom GPUs (GPUs not visible in the container)
4. Missing GPU entries (fewer GPUs than visible)
5. Wrong PCIe hierarchy (incorrect switch topology)

**Deliverable**: Robustness analysis — defines requirements for daemon topology
accuracy. Characterizes how topology errors affect transport selection and bandwidth.

## Risk

NCCL may cross-validate `NCCL_TOPO_FILE` against `/sys/bus/pci` entries. If the
XML claims NVLink connections to GPUs not visible in the container's sysfs, NCCL
might reject the topology. Mitigation: sysfs bind-mounts (selective
`/sys/bus/pci/devices/` entries from host) or FUSE-based sysfs overlay.

This would itself be an important finding — it means topology virtualization
requires sysfs virtualization as well, increasing the interception surface.

## Results Directory

```
results/
├── nccl_topo_baremetal.xml    # NCCL auto-detected topology (reference)
├── nccl_topo_generated.xml    # Daemon-generated topology
├── graph_baremetal.txt        # NCCL transport graph (bare metal)
├── graph_shim_only.txt        # NCCL transport graph (shim, no topo file)
├── graph_shim_topo.txt        # NCCL transport graph (shim + topo file)
├── bw_comparison.json         # Bandwidth comparison across configs
├── xml_schema.md              # NCCL topology XML schema documentation
└── resilience_tests.json      # Incomplete/incorrect topology test results
```
