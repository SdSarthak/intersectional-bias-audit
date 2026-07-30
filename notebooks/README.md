# Notebooks

## `intersectional_audit.ipynb` — start here

A short walkthrough that drives the `bias_audit` package: load the data, train, audit every
subgroup, and compare the intersectional view with the single-attribute one. This is the notebook to
run.

## `main.ipynb`, `final.ipynb` — original Colab exports, kept for provenance

These are the notebooks the paper was drafted from, with their outputs stripped. **They are superseded
by the `bias_audit` package and should not be used to produce results.** Known problems, all fixed in
the package:

* The plotting cells fall back to invented constants (`locals().get('spd_gender', 0.05)`) and to
  `np.random.choice(...)` demographics whenever an expected variable is missing, so several charts do
  not describe the model being audited.
* One visualisation cell overwrites the real predictions with a six-row dummy dataset before
  plotting.
* The `generalized_fnr` / `generalized_fpr` calls do not match the installed AIF360 signature; the
  exceptions are swallowed, which is why the FNR/FPR columns in the original
  `intersectional_results.csv` are empty.
* Protected attributes are rebuilt by hand after several `reset_index` calls and drift out of
  alignment with the predictions.
* Age bin edges of `[0, 30, 50, 150]` with right-closed intervals place 30-year-olds in the
  `Young (<30)` band.

See the "Notes on reproducing the paper" section of the top-level [README](../README.md).
