# Reproducing the Best Leaderboard Submission

This repository reproduces the team's best public leaderboard submission for Assignment 4.

Best submission file produced by these steps:

```text
submission_combined_best.zip
```

Best public leaderboard score:

```text
0.444834
```

## 1. Repository contents needed

The following files must be present in the repository root:

```text
forge_v3.py
forge_v7_avgmix.py
swap_wm_group.py
submission.py
README.md
```

The dataset must be extracted as:

```text
Dataset/
├── clean_targets/
│   ├── 1.png
│   ├── ...
│   └── 200.png
└── watermarked_sources/
    ├── WM_1/
    ├── WM_2/
    ├── WM_3/
    ├── WM_4/
    ├── WM_5/
    ├── WM_6/
    ├── WM_7/
    └── WM_8/
```

## 2. Environment setup

Create and activate a fresh Python environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy==1.26.4 pillow opencv-python
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy==1.26.4 pillow opencv-python
```

## 3. Generate the base average-mix submission

Run:

```powershell
python .\forge_v7_avgmix.py --dataset .\Dataset --avg-mode mean --lambda-avg 0.16 --residual-alpha 2.0 --low-clip 28 --total-clip 36 --texture 0.15 --out submission_v7_avgmix_l016_r2.zip --work-dir tmp_v7_avgmix_l016_r2
```

This produces:

```text
submission_v7_avgmix_l016_r2.zip
```

## 4. Generate the stronger WM_1 variant

Run:

```powershell
python .\forge_v7_avgmix.py --dataset .\Dataset --avg-mode mean --lambda-avg 0.20 --residual-alpha 1.0 --low-clip 32 --total-clip 42 --texture 0.10 --out submission_v7_avgmix_l020_r1.zip --work-dir tmp_v7_avgmix_l020_r1
```

This produces:

```text
submission_v7_avgmix_l020_r1.zip
```

## 5. Create the final best submission

Replace only the `WM_1` target batch, images `1.png` to `25.png`, from the stronger variant into the base submission:

```powershell
python .\swap_wm_group.py --base submission_v7_avgmix_l016_r2.zip --variant submission_v7_avgmix_l020_r1.zip --wm 1 --out submission_combined_best.zip
```

The final file to submit is:

```text
submission_combined_best.zip
```

It must contain exactly:

```text
1.png
2.png
...
200.png
```

with no subfolders and no renamed files.

## 6. Submit to the leaderboard

In `submission.py`, set:

```python
API_KEY = "YOUR_TEAM_API_KEY"
FILE_PATH = Path("submission_combined_best.zip")
SUBMIT = True
```

Then run:

```powershell
python .\submission.py
```
