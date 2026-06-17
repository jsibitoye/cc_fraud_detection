# UI Report QC Checklist

**Document:** UI_Fraud_Detection_Final_Report.docx / .pdf
**Title:** Progressive Feature Engineering and Imbalanced Machine Learning for Credit Card Fraud Detection
**Student:** Joshua Ibitoye (Matric No: 225791), M.Sc. Information Security, University of Ibadan
**Date of QC:** 17 June 2026
**Rendered length:** 50 pages (PDF)

## Forbidden content checks

| Check | Result | Status |
|---|---|---|
| No "Nigeria-style" | 0 occurrences | PASS |
| No "text mining" | 0 occurrences | PASS |
| No "[REF]" | 0 occurrences | PASS |
| No "TODO" | 0 occurrences | PASS |
| No "citation needed" | 0 occurrences | PASS |
| No "placeholder text" / lorem / filler | 0 occurrences | PASS |
| Only permitted bracket placeholders ([Department], [Faculty], [Supervisor Name], [Month, Year]) | Present, as required | PASS |

## Content-integrity checks

| Check | Result | Status |
|---|---|---|
| No unsupported real-bank-data claim | Data described throughout as synthetic/researcher-generated | PASS |
| No real-time / deployment claim | Explicitly disclaimed in Scope, Ethics and Limitations | PASS |
| Raw merged file not treated as a sixth experiment | Mentioned only as cleaning/provenance background | PASS |
| No false universal-SMOTE claim | SMOTE benefit framed as default-threshold only; equalised after tuning | PASS |
| No false v5-superiority claim | v3–v5 stated as practically tied throughout | PASS |
| Synthetic data, no external validation, leakage-risk proxies disclosed | Stated in Abstract, 1.7, 1.8, 3.13, Ch.4, Ch.5 | PASS |
| Results match supplied evidence | Tables 4.2–4.4 and Appendix C reproduce supplied final metrics | PASS |
| Best tuned model reported accurately | v5 No-SMOTE Random Forest, threshold 0.80, F1 ≈ 0.862 | PASS |

## Structure and numbering checks

| Check | Result | Status |
|---|---|---|
| Front matter complete (flyleaf, title, certification, dedication, acknowledgements, abstract, abbreviations, TOC, LoT, LoF) | All present | PASS |
| All five chapters present with required sections | Chapters 1–5 complete | PASS |
| References section present | 32 references | PASS |
| Appendices A–E present | All present | PASS |
| Tables numbered by chapter | Table 2.1, 3.1–3.3, 4.1–4.5 (+ A.1, C.1) | PASS |
| Figures numbered by chapter | Fig. 2.1, 3.1, 3.2, 4.1–4.8 | PASS |
| Every table/figure introduced in text before it appears | Verified for all | PASS |
| Table captions above; figure captions below | Verified in render | PASS |
| Table of Contents populated with page numbers | Auto-measured, dot leaders | PASS |
| List of Tables populated with page numbers | Auto-measured | PASS |
| List of Figures populated with page numbers | Auto-measured | PASS |
| Page numbers present (roman front matter, arabic body) | Verified in render | PASS |

## Abstract checks

| Check | Result | Status |
|---|---|---|
| Four paragraphs | Yes | PASS |
| Under 500 words | ~342 words | PASS |
| 3–5 keywords after abstract | 5 keywords | PASS |
| Includes problem, aim, methodology, dataset size/ratio, models, SMOTE comparison, threshold tuning, best default & tuned results, key finding, limitations | All present | PASS |

## Referencing checks

| Check | Result | Status |
|---|---|---|
| UIMS-consistent author–date style | Applied consistently | PASS |
| Every reference cited in text | 32/32 cited (Pedregosa cited in Appendix D) | PASS |
| No uncited source in References | Verified | PASS |
| Real sources only (no invented citations) | Verified | PASS |

## Formatting checks

| Check | Result | Status |
|---|---|---|
| Times New Roman, 12 pt body | Applied via Normal style | PASS |
| Main headings ≤ 14 pt | H1 = 14 pt, H2 = 13 pt, H3 = 12 pt | PASS |
| A4 page size | 21.0 × 29.7 cm | PASS |
| Body line spacing 1.5 | Applied | PASS |
| Left margin 3.7 cm; top/right/bottom 2.5 cm | Applied | PASS |
| Justified body text | Applied | PASS |
| British spelling | Applied (US spellings appear only inside verbatim reference titles) | PASS |
| DOCX created | UI_Fraud_Detection_Final_Report.docx | PASS |
| PDF created | UI_Fraud_Detection_Final_Report.pdf (50 pp) | PASS |

## Outstanding (administrative, for the student)

- Fill front-matter placeholders: Department, Faculty, Supervisor Name, Month/Year.
- Complete certification signatures and dates after supervisor review.
- Complete the dedication/acknowledgements personalisation if desired.

**Overall QC verdict:** PASS (administrative metadata placeholders excepted).
