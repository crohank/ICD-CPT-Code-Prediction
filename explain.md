# ICD-10 Code Prediction from Hospital Notes

## What is this project?

When you visit a hospital, doctors write detailed notes about your visit — what symptoms you had, what tests were run, what treatments were given. Before you leave, someone has to assign **medical billing codes** (called ICD-10 codes) to your visit. These codes tell insurance companies what happened so the hospital gets paid.

There are over **70,000** of these codes. Assigning them manually is slow, expensive, and error-prone. Hospitals employ dedicated staff called medical coders just for this task.

**Our project builds a system that reads a doctor's note and automatically suggests the correct billing codes.** Think of it like autocomplete, but for medical billing.

---

## What data are we using?

We use **MIMIC-IV**, a real (but de-identified) dataset of hospital records from Beth Israel Deaconess Medical Center in Boston. It contains over **120,000 discharge summaries** — the notes written when a patient leaves the hospital.

Each note is paired with the billing codes that were actually assigned by human coders. We use this to teach our system what codes go with what kind of notes.

We focus on the **Top 50 most common codes**, which cover conditions like diabetes, heart failure, high blood pressure, kidney disease, and pneumonia.

---

## How does it work?

We built **three different approaches** and combined them:

### Model A — Keyword Matching
The simplest approach. It looks at which words appear in a note and how often. If a note mentions "insulin", "glucose", and "hyperglycemia" frequently, it predicts diabetes codes. This is like a very sophisticated version of searching for keywords.

**Result:** Got the right codes ~60% of the time.

### Model B — Reading Comprehension (AI)
Uses a pre-trained AI model called **ClinicalBERT** that was specifically trained to understand medical language. Instead of just counting words, it understands context — for example, that "the patient denied chest pain" means the patient does NOT have chest pain, even though the words "chest pain" appear.

The limitation: it can only read ~500 words at a time, but hospital notes are often 2,000+ words long. So it misses important information at the end of long notes.

**Result:** ~52% accuracy (worse than keyword matching because it loses information from long notes).

### Model C — The Full Picture (Advanced AI)
Our most sophisticated approach. It solves Model B's length problem by:
1. **Splitting** long notes into overlapping pieces (like reading a book chapter by chapter, with some overlap so you don't lose context between chapters)
2. **Reading** each piece with ClinicalBERT
3. **Paying attention** to the parts that matter for each specific code — when predicting a heart failure code, it focuses on cardiac-related sentences; when predicting a diabetes code, it focuses on metabolic-related sentences

**Result:** Currently being retrained with bug fixes. Expected ~58-62% accuracy.

### Ensemble — Best of All Worlds
We combine Model A and Model C's predictions by averaging them. Model A is good at common patterns; Model C is good at understanding nuance. Together, they cover each other's weaknesses.

**Result:** ~62.5% accuracy (our best so far), expected to reach ~65-68% after current improvements.

---

## How good is this?

| What we measure | Our score | What it means |
|---|---|---|
| Micro-F1 = 0.625 | 62.5% | On average, our predictions are right about 6 out of 10 times |
| AUROC = 0.933 | 93.3% | The system is very good at ranking which codes are likely vs unlikely |

For context, published research papers on the same task report scores in the 55-65% range. Our ensemble is competitive with the state of the art.

---

## What else did we build?

Beyond the AI models, we built a complete production-ready system:

- **API Server** — Other software can send a doctor's note and get back predicted codes (like how Google Maps has an API that other apps use for directions)
- **Web Demo** — A simple website where you can paste a note and see predictions with explanations
- **Docker Setup** — Everything packaged so it runs the same way on any computer (like shipping furniture in a standardized container)
- **Automated Tests** — Code that checks our system still works correctly after every change

---

## What's next?

We're currently retraining Model C with several improvements:
- **Better loss function** (focal loss) that helps the system learn rare codes better
- **Semantic initialization** — instead of starting from scratch, we tell the system what each code means in plain English
- **Probability calibration** — making the system's confidence scores more trustworthy
- **Per-code thresholds** — different codes need different confidence levels to trigger a prediction

After retraining, we expect the ensemble to reach **65-68% accuracy**.
