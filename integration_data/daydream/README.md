# Daydream 1 LAS and DLIS integration fixtures

These full-size files support the real DLIS-reader and FMI integration tests.
They are stored with Git LFS and are intentionally excluded from the Python
wheel and source distribution. Run `git lfs pull` after cloning before running
the integration tests or the FMI notebook.

The files are public Queensland well-completion data for Daydream 1, ATP 645.
The repository filenames are normalized for portability; the original delivery
filenames and checksums are recorded below.

| Repository file | Original delivery file | Bytes | SHA-256 |
|---|---|---:|---|
| `daydream1_fmi_processed_dynamic_image.dlis` | `Daydream 1_FMI_Processed_Dynamic_Image.dlis` | 111,779,520 | `1449d86411a22299d6bed85accb1e44259c7fd02594f154fcf0dd7598ba27b75` |
| `daydream1_deep_pex_aps_ppc_ecs.dlis` | `QGC_Daydream-1_Dual-PEX-APS-PPC-ECS.dlis` | 172,587,816 | `4e1943402d5abe7ce81a10bdefd8cd8c0a31c2a2b2bc22a0ef91997b3671af86` |
| `daydream1_deep_pex_aps_ppc_hngs_ecs.las` | `QGC_Daydream-1_Dual-PEX-APS-PPC-HNGS-ECS_Cust.las` | 395,884,528 | `cc63cc99bcdfa12af4580d7993a0f0e530b7395180bc29ec3b55ab1a07914a42` |
| `daydream1_shallow_aps_pex_hngs.dlis` | `QGC_Daydream1_R2-APS-PEX-HNGS_1860-835_Main.dlis` | 120,929,868 | `1285f048735bde98e9f254b0c8e6d43187eb303f900de4e83f11df45dca4b1df` |
| `daydream1_shallow_aps_pex_hngs.las` | `Daydream 1_Suite 1_Run 2_APS-PEX-HNGS.las` | 30,031,316 | `27c96207fec5296aa56dd2222ae1511412bd6fe7d5731eeab843e27dd86e972d` |

The larger raw FMI acquisition file is not included because the processed FMI
fixture above is the file consumed by the current image workflow.

## LAS/DLIS equivalence validation

The shallow and deep LAS/DLIS pairs are matching deliveries from the same
logging runs. Run the paired reader and dynamic-workflow validations with:

```powershell
python validation/las_dlis_equivalence.py --preset shallow --output build/daydream_las_dlis_equivalence.json
python validation/las_dlis_equivalence.py --preset deep --output build/daydream_deep_las_dlis_equivalence.json
```

The report separates two questions:

- `native_delivery` runs every applicable workflow step on each file as
  delivered. The DLIS may contain additional vendor-processed curves that are
  intentionally preserved and can therefore change the active interpretation.
- `matched_common_channels` uses the exact standard-resolution physical curves
  present in both files. This isolates file-format handling from differences in
  delivery content and sampling.

The paired acquisition has invaded-zone resistivity but no calibrated deep
resistivity. Saturation-dependent steps are therefore expected to be skipped in
both workflows. The raw uncalibrated DLIS resistivity channel remains available
under its vendor mnemonic and is not promoted to `RDEP`.

The deep DLIS includes calibrated deep resistivity (`RVDR`), while its matching
LAS contains only the uncalibrated `RVDRU` delivery. Consequently, the native
DLIS workflow can run saturation and net pay while the native LAS workflow
correctly skips them. The matched-common-channel comparison excludes both
calibrated and uncalibrated deep resistivity so that it tests only equivalent
inputs. Uncalibrated resistivity is still compared as a raw transport check but
is never promoted to an interpretation curve.
