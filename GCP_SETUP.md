# GCP Setup Guide — ICD-10 Code Prediction (MIMIC-IV)

---

## Which approach should I use?

| | **Free Colab + GCP VM** ← *you are here* | Colab Pro |
|---|---|---|
| Cost | ~$5–15 from your $300 GCP credits | $9.99/mo |
| GPU | T4 on a GCP VM you control | T4 (not always guaranteed on free tier) |
| Session limit | **None** — VM stays up as long as you want | 24h |
| Setup effort | ~30 min one-time setup | Instant |
| Code changes needed | **None** | None |

**How it works:** You run a Jupyter server on a GCP VM (with a T4 GPU). You SSH-tunnel port 8888 to your laptop. Then in free Colab you click **Runtime → Connect to a custom runtime** and point it at `localhost:8888`. The Colab UI lives in your browser for free; all computation runs on the GCP VM and is billed against your $300 credits.

`google.colab.drive.mount()` still works — it mounts your Google Drive onto the VM at `/content/drive`. GCS auth still works. **The notebooks run completely unchanged.**

```
Your browser (free Colab UI)
        │  SSH tunnel (port 8888)
        ▼
GCP VM: n1-standard-8 + T4 GPU   ← compute charged to your $300 credits
        │  google.colab.drive.mount()
        ▼
Google Drive: MyDrive/mimic_icd/  ← outputs saved here
        │  gcsfs
        ▼
PhysioNet GCS: gs://physionet-data/  ← MIMIC-IV raw data
```

---

## Prerequisites

| Item | Purpose |
|------|---------|
| Google account (personal) | GCP project billing, Google Drive storage |
| Google account (school/PhysioNet email) | Access `gs://physionet-data/` |
| PhysioNet credentialed access to MIMIC-IV + MIMIC-IV-Note | Raw data |
| GCP free trial activated ($300 credits) | Compute billing |

> Your personal and school accounts can be the same if your school email is your PhysioNet-registered email.

---

## Part 1: Activate GCP Free Credits

1. Go to **console.cloud.google.com** and sign in with your **personal** Google account
2. Click **"Activate free trial"** — you get **$300 credits** valid for 90 days
3. You must add a credit card to activate (you won't be charged unless you manually upgrade)
4. Once activated, create a new project: **Navigation menu → New Project** → name it `mimic-icd` (or anything)

> **Note:** Free trial accounts have GPU quota = 0 by default. You need to request GPU quota. Do this now — it can take a few minutes to a few hours to be approved.

**Request GPU quota:**
1. Go to **IAM & Admin → Quotas**
2. Search for `NVIDIA T4 GPUs` in your region (use `us-central1` or `us-east1` — cheapest)
3. Select it → **Edit Quotas** → request **1** GPU → submit
4. You'll get an email when approved (usually instant to a few hours)

---

## Part 2: Choose Your GPU

All GPUs below run the notebooks without code changes. The batch-size auto-detection handles all of them.

| GPU | VRAM | Machine type | Price/hr | Model B batch | Top-50 train | Top-500 train | Total cost* |
|-----|------|-------------|---------|--------------|-------------|--------------|-------------|
| **T4** | 16 GB | `n1-standard-8` | ~$0.55 | 16 | ~90 min | ~4 hrs | ~$2 |
| **L4** ← recommended | 24 GB | `g2-standard-8` | ~$0.85 | 32 | ~55 min | ~2.5 hrs | ~$2.50 |
| **A100 40GB** | 40 GB | `a2-highgpu-1g` | ~$3.67 | 64 | ~25 min | ~1 hr | ~$8 |
| **A100 80GB** | 80 GB | `a2-ultragpu-1g` | ~$5.50 | 128 | ~18 min | ~45 min | ~$12 |
| V100 | 16 GB | `n1-standard-8` | ~$2.48 | 16 | ~85 min | ~4 hrs | ~$8 | ← skip, not worth it |

*Total cost = notebooks 01–05, both Top-50 and Top-500 runs

**Recommendation: L4 (`g2-standard-8`)**
- 50% more VRAM than T4 → batch_size=32 instead of 16 → nearly 2x faster
- Only ~$0.30/hr more than T4
- Full pipeline costs ~$2.50 from your $300 credits
- Available in `us-central1-a`, `us-east1-b`, `europe-west4-a`

**Use A100 if:** you want fast iteration (e.g., tuning hyperparameters, re-running multiple times). Still only ~$8 total from $300.

**Skip V100:** same VRAM as T4 but 4-5x more expensive. Not worth it.

> **GPU quota:** Before creating a VM, check you have quota for the GPU type. Go to **IAM & Admin → Quotas**, search for the GPU name in your target region. If quota = 0, request an increase — usually approved in minutes for T4/L4.

## Part 3: Create the GCP VM

Use the **Deep Learning VM** image — it comes with CUDA, PyTorch, and JupyterLab pre-installed, saving ~1 hour of manual setup.

### 3.1 Create the instance via Cloud Console

1. Go to **Compute Engine → VM instances → Create instance**
2. Fill in based on your chosen GPU:

**For L4 (recommended):**

| Field | Value |
|-------|-------|
| Name | `mimic-gpu` |
| Region | `us-central1` |
| Zone | `us-central1-a` |
| Machine type | `g2-standard-8` (8 vCPU, 32 GB RAM) |
| GPU | Already included in g2 machines — shows 1x L4 automatically |
| Boot disk | Click **Change** → **Deep Learning on Linux** → `PyTorch 2.x + CUDA 12.x` → 100 GB SSD |
| Firewall | ✅ Allow HTTP / ✅ Allow HTTPS |

**For A100 40GB:**

| Field | Value |
|-------|-------|
| Machine type | `a2-highgpu-1g` (12 vCPU, 85 GB RAM, 1x A100 40GB — GPU included) |
| Region/Zone | `us-central1-c` or `us-east1-b` (A100 availability varies) |
| Boot disk | Same Deep Learning VM image |

3. Click **Create** — VM starts in ~2 min

### 3.2 (Optional) Use gcloud CLI

**L4:**
```bash
gcloud compute instances create mimic-gpu \
  --zone=us-central1-a \
  --machine-type=g2-standard-8 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE
```

**A100 40GB:**
```bash
gcloud compute instances create mimic-gpu \
  --zone=us-central1-c \
  --machine-type=a2-highgpu-1g \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE
```

**T4 (if L4 quota unavailable):**
```bash
gcloud compute instances create mimic-gpu \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True"
```

---

## Part 4: One-Time VM Setup (~15 min)

### 4.1 SSH into the VM

**Option A — Cloud Console (easiest):**
Compute Engine → VM instances → click **SSH** button next to your VM. A browser terminal opens.

**Option B — gcloud CLI (from your local terminal):**
```bash
gcloud compute ssh mimic-gpu --zone=us-central1-a
```

### 4.2 Verify GPU is ready

```bash
nvidia-smi
```
Should show a T4 GPU. If not, run `sudo /opt/deeplearning/install-driver.sh` and reboot.

### 4.3 Install the Colab custom runtime extension

```bash
pip install jupyter_http_over_ws
jupyter server extension enable --py jupyter_http_over_ws
```

### 4.4 Authenticate with PhysioNet GCS

This is a one-time auth on the VM. Run this in the SSH terminal:

```bash
gcloud auth application-default login --no-launch-browser
```

A URL is printed. Open it in your browser, sign in with your **PhysioNet/school account**, and paste the verification code back into the terminal. Done — credentials are saved to the VM permanently.

### 4.5 Clone the repo

```bash
cd ~
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git icd-project
cd icd-project
git checkout v2
pip install -r requirements.txt
```

---

## Part 5: Connect Free Colab to the GCP VM

Do this every time you want to work (takes ~1 min).

### Step 1 — Start Jupyter on the VM

In your SSH terminal on the VM:

```bash
jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port=8888 \
  --no-browser \
  --NotebookApp.disable_check_xsrf=True
```

You'll see output like:
```
[I] Serving notebooks from local directory: /home/user/icd-project
[I] http://localhost:8888/?token=abc123def456...
```

**Copy the token** (the long string after `token=`). You'll need it.

### Step 2 — Open an SSH tunnel on your laptop

Open a **new terminal** on your local machine (not the VM):

```bash
gcloud compute ssh mimic-gpu --zone=us-central1-a -- -L 8888:localhost:8888 -N
```

Leave this terminal open. It forwards VM port 8888 to your laptop port 8888.

> **Windows users:** If gcloud isn't installed, use the Cloud Console SSH button and then use the built-in port forwarding:  
> Cloud Console → SSH → gear icon → **Port forwarding** → add `8888 → localhost:8888`

### Step 3 — Connect free Colab to the VM

1. Open [colab.research.google.com](https://colab.research.google.com)
2. Open a notebook (from GitHub: File → Open → GitHub → your repo → branch `v2` → `notebooks/01_data_extraction.ipynb`)
3. Top-right corner: click the **dropdown arrow** next to "Connect"
4. Click **"Connect to a local runtime"**
5. Enter: `http://localhost:8888/?token=abc123def456...` (paste your full token URL)
6. Click **Connect**

The indicator turns green and shows **RAM/Disk bars** — that's your GCP VM's resources.

> You can now open all 5 notebooks this way, each in a separate Colab tab, all connected to the same VM.

---

## Part 6: Run the Notebooks

The notebooks in `notebooks/` run **completely unchanged**. The Drive mount works (mounts to the VM), GCS auth works (already done in Step 3.4). Run them in order:

### Notebook 01 — Data Extraction (~30 min)

Open `notebooks/01_data_extraction.ipynb` in Colab (connected to VM).

**Cell 2 (Drive + GCS auth):**
- `drive.mount('/content/drive')` — mounts your Google Drive on the VM. A Colab popup will appear asking you to authorize. Sign in with your **personal** account.
- `gcloud auth application-default login --no-launch-browser` — since you already authenticated in Part 3 Step 4, this will either skip or confirm existing credentials. If it prompts again, just paste the verification code as before.

**Expected output:**
```
Cohort size (note + ICD-10 codes): ~122,000
Vocab (freq>=10): 7940
train: 85437 rows  val: 18195 rows  test: 18672 rows
All splits saved to /content/drive/MyDrive/mimic_icd/datasets
```

### Notebook 02 — Preprocessing (~10 min)

Open `notebooks/02_preprocessing.ipynb`. **Cell 4 config:**
```python
TOP_K_LABELS = 50        # 50 = fast + standard benchmark; 500 = extended
TFIDF_MAX_FEATURES = 50_000
```
Expected: `Y_train shape: (85437, 50)`

### Notebook 03 — Model A (~2–5 min for Top-50)

Open `notebooks/03_model_a_tfidf_baseline.ipynb`. Cell 1 attempts to install cuML. On the Deep Learning VM with CUDA, cuML installation is more likely to succeed than on free Colab.

```
Training OvR on 85437 samples × 50 labels (GPU=True)...
Done.  micro_f1: 0.xxxx
```

### Notebook 04 — Model B (~1–2 hours for Top-50)

Open `notebooks/04_model_b_transformer.ipynb`. **Verify GPU is detected** in Cell 4:
```
GPU: Tesla T4, 14.7 GB VRAM
Device: cuda  Batch: 16  GradAccum: 2  Effective: 32
```

Training saves mid-epoch checkpoints to Drive every 2000 steps. If your SSH tunnel drops, the VM keeps running — just reconnect the tunnel and Colab, and the training is still going on the VM.

**To resume after a disconnect:**
```python
# Add before the training loop to resume from checkpoint
ckpt = torch.load(f'{MODEL_DIR}/checkpoint_e2_s4000.pt', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])
```

### Notebook 05 — Evaluation (~5–10 min)

Open `notebooks/05_evaluation_demo.ipynb`. Produces the final comparison table.

---

## Part 7: Managing the VM (Important — Avoid Wasted Credits)

### Stop the VM when not training
**Compute Engine → VM instances → select `mimic-gpu` → Stop**

A stopped VM costs only ~$0.04/hr for disk storage. A running VM with T4 costs ~$0.55/hr.

### Start it back up
**Compute Engine → VM instances → select `mimic-gpu` → Start**

Then re-run Part 4 (start Jupyter, open SSH tunnel, connect Colab). Your Drive data persists — no re-running notebooks 01-02 needed.

### Check remaining credits
**Billing → Credits** in the Cloud Console. Your $300 runs out after 90 days even if unused.

---

## Part 8: Cost Breakdown

| Task | Duration | Cost (standard) | Cost (preemptible*) |
|------|----------|----------------|---------------------|
| Notebook 01 — extraction | ~30 min | ~$0.28 | ~$0.07 |
| Notebook 02 — preprocessing | ~15 min | ~$0.14 | ~$0.04 |
| Notebook 03 — Model A (Top-50) | ~5 min | ~$0.05 | ~$0.01 |
| Notebook 04 — Model B (Top-50) | ~2 hours | ~$1.10 | ~$0.28 |
| Notebook 04 — Model B (Top-500) | ~5 hours | ~$2.75 | ~$0.70 |
| **Total (Top-50 run)** | ~3 hours | **~$1.60** | **~$0.40** |
| **Total (Top-500 run)** | ~6 hours | **~$3.30** | **~$0.82** |

*Preemptible VMs can be stopped by Google at any time (rare). Use standard instances for training jobs since the VM keeps running even if your laptop closes.

**Your $300 credits can run this pipeline ~90–180 times.** Cost is not a concern.

---

## Part 9: Troubleshooting

### "Could not connect to local runtime"
- SSH tunnel has dropped — rerun the tunnel command in Part 4 Step 2
- Jupyter server stopped — SSH into VM and restart it (Part 4 Step 1)

### Drive doesn't mount on the VM
If `drive.mount()` fails or doesn't show a popup, run this in a Colab cell:
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### "403 Forbidden" on GCS
The PhysioNet credentials on the VM have expired. SSH into the VM and re-run:
```bash
gcloud auth application-default login --no-launch-browser
```

### GPU not detected in notebooks
SSH into VM and check:
```bash
nvidia-smi     # should show Tesla T4
python -c "import torch; print(torch.cuda.is_available())"  # should print True
```
If `nvidia-smi` fails: `sudo /opt/deeplearning/install-driver.sh` then reboot the VM.

### "CUDA out of memory" in Notebook 04
Override the auto-detected batch size in Cell 4:
```python
BATCH_SIZE = 8
GRAD_ACCUM = 4
```

### VM shows "$0.00 credits" being charged
Confirm you're using the project connected to the free trial. Check **Billing → Credits** to see the balance.

---

## Part 10: Run Order & Time Summary

```
VM running + SSH tunnel open + Colab connected
        │
        ▼
Notebook 01  (~30 min)  →  cohort parquets saved to Drive
        │
        ▼
Notebook 02  (~15 min)  →  TF-IDF features + label matrices saved to Drive
        │
        ├─▶  Notebook 03  (~5 min)   →  Model A results
        │
        └─▶  Notebook 04  (~2 hrs)   →  Model B checkpoint + results
                               │
                               ▼
                         Notebook 05  (~10 min)  →  final_comparison.csv
```

Notebooks 03 and 04 are independent — you and your teammate can run them in separate Colab tabs connected to the same VM simultaneously.

---

## Quick Reference

| Item | Value |
|------|-------|
| VM name | `mimic-gpu` |
| Zone | `us-central1-a` |
| Jupyter port | `8888` |
| Drive output path | `/content/drive/MyDrive/mimic_icd/` |
| Start Jupyter on VM | `jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --no-browser --NotebookApp.disable_check_xsrf=True` |
| SSH tunnel command | `gcloud compute ssh mimic-gpu --zone=us-central1-a -- -L 8888:localhost:8888 -N` |
| Stop VM | Cloud Console → Compute Engine → Stop |
| Check GCP billing | Cloud Console → Billing → Credits |
