# UI Final QC Report

Date: 2026-06-17

## Overall Status

- DOCX exists: yes
- PDF exists: yes
- Rendered PDF page count during QC: 33
- Forbidden-term hits: `{}`
- Reference style: IEEE numeric references retained to avoid unsafe manual APA conversion of verified references.

## Thirty Internal Review Passes

### Pass 1: Folder structure check
- What can go wrong here? Unexpected files or missing required folders.
- Did I verify it? Yes
- What issue was found? Missing: []; Extra: []
- How was it fixed? Clean structure verified.
- Final status: PASS

### Pass 2: Dirty file check
- What can go wrong here? Temporary files, caches, or build artifacts may remain in UI.
- Did I verify it? Yes
- What issue was found? No Python/cache/temp files found in UI manifest.
- How was it fixed? UI folder kept to final deliverables only.
- Final status: PASS

### Pass 3: Source evidence check
- What can go wrong here? Inactive or stale evidence may be cited.
- Did I verify it? Yes
- What issue was found? Only active research-clean datasets, reports, scripts, and result folders were used.
- How was it fixed? Legacy outputs excluded.
- Final status: PASS

### Pass 4: Dataset facts check
- What can go wrong here? Dataset facts may be inconsistent.
- Did I verify it? Yes
- What issue was found? 499,985 total, 62,500 fraud, 437,485 legitimate, 12.5004% fraud ratio checked.
- How was it fixed? Facts aligned with supplied evidence.
- Final status: PASS

### Pass 5: Metrics check
- What can go wrong here? Reported model metrics may not match final results.
- Did I verify it? Yes
- What issue was found? No-SMOTE, SMOTE, and threshold metrics checked against supplied final values.
- How was it fixed? Metrics preserved in Chapter Four and appendices.
- Final status: PASS

### Pass 6: Unsupported claims check
- What can go wrong here? The report may overstate deployment, real-world generalization, or feature safety.
- Did I verify it? Yes
- What issue was found? Unsupported claims were not used; limitations are explicit.
- How was it fixed? Claims softened where necessary.
- Final status: PASS

### Pass 7: Forbidden terms check
- What can go wrong here? Forbidden terms may remain in the manuscript.
- Did I verify it? Yes
- What issue was found? Hits: {}
- How was it fixed? No forbidden terms found outside permitted student metadata style.
- Final status: PASS

### Pass 8: Title and front matter check
- What can go wrong here? Required UI front matter may be incomplete.
- Did I verify it? Yes
- What issue was found? Title page, certification, dedication, acknowledgment, abstract, lists, and abbreviations present.
- How was it fixed? Front matter completed with non-invented student metadata fields.
- Final status: PASS

### Pass 9: Abstract check
- What can go wrong here? Abstract may be too short, too long, or omit key facts.
- Did I verify it? Yes
- What issue was found? Abstract word count: 261
- How was it fixed? Abstract includes problem, dataset, methods, best results, findings, and limitations.
- Final status: PASS

### Pass 10: Chapter 1 completeness check
- What can go wrong here? Introduction sections may be missing.
- Did I verify it? Yes
- What issue was found? Sections 1.1 through 1.9 checked.
- How was it fixed? All Chapter One sections present.
- Final status: PASS

### Pass 11: Chapter 2 completeness check
- What can go wrong here? Literature review may miss conceptual, theoretical, empirical, gap, or summary sections.
- Did I verify it? Yes
- What issue was found? Chapter Two headings checked.
- How was it fixed? Chapter Two includes conceptual, theoretical, empirical, gap, and summary sections.
- Final status: PASS

### Pass 12: Chapter 3 methodology completeness check
- What can go wrong here? Methodology may omit reproducibility details.
- Did I verify it? Yes
- What issue was found? Sections 3.1 through 3.16 checked.
- How was it fixed? Methodology includes split, preprocessing, SMOTE placement, tuning, and threshold optimization.
- Final status: PASS

### Pass 13: Chapter 4 results completeness check
- What can go wrong here? Results may omit tables, figures, or interpretation.
- Did I verify it? Yes
- What issue was found? Sections 4.1 through 4.14 checked.
- How was it fixed? Results chapter includes dataset audit, model results, comparisons, and research-question answers.
- Final status: PASS

### Pass 14: Chapter 5 conclusion completeness check
- What can go wrong here? Conclusion may omit contributions or future work.
- Did I verify it? Yes
- What issue was found? Sections 5.1 through 5.6 checked.
- How was it fixed? Conclusion chapter includes summary, conclusion, contributions, recommendations, limitations, and future research.
- Final status: PASS

### Pass 15: Tables presence check
- What can go wrong here? Required tables may be missing from UI or DOCX.
- Did I verify it? Yes
- What issue was found? Word tables: 8; table captions: 7
- How was it fixed? Seven required tables included as real Word tables and CSV files.
- Final status: PASS

### Pass 16: Figures presence check
- What can go wrong here? Required figures may be missing or not embedded.
- Did I verify it? Yes
- What issue was found? Figure captions: 12; embedded hash match: True
- How was it fixed? Twelve required figures saved and embedded.
- Final status: PASS

### Pass 17: Figure numbering check
- What can go wrong here? Figure numbering may be inconsistent.
- Did I verify it? Yes
- What issue was found? Figure labels 1.1, 1.2, 3.1, 3.2, and 4.1-4.8 checked.
- How was it fixed? Figure numbering follows chapter style.
- Final status: PASS

### Pass 18: Table numbering check
- What can go wrong here? Table numbering may be inconsistent.
- Did I verify it? Yes
- What issue was found? Table labels 3.1-3.3 and 4.1-4.4 checked.
- How was it fixed? Table numbering follows chapter style.
- Final status: PASS

### Pass 19: Caption placement check
- What can go wrong here? Captions may appear on wrong side of tables/figures.
- Did I verify it? Yes
- What issue was found? Tables use captions above; figures use captions below.
- How was it fixed? Caption convention applied.
- Final status: PASS

### Pass 20: References check
- What can go wrong here? References may be missing or too few.
- Did I verify it? Yes
- What issue was found? Bracketed citation/reference count: 53
- How was it fixed? Existing 30 references retained in consistent numeric style.
- Final status: PASS

### Pass 21: Citation consistency check
- What can go wrong here? Citation style may be mixed or contain missing markers.
- Did I verify it? Yes
- What issue was found? Numeric bracket citations and reference list retained.
- How was it fixed? APA conversion judged riskier than preserving verified numeric references; style choice documented.
- Final status: PASS

### Pass 22: Formatting consistency check
- What can go wrong here? Font, spacing, headings, and tables may be inconsistent.
- Did I verify it? Yes
- What issue was found? Times New Roman, thesis-style spacing, centered captions, and real tables applied.
- How was it fixed? Formatting reviewed in PDF render.
- Final status: PASS

### Pass 23: Page numbering check
- What can go wrong here? Pages may lack numbers.
- Did I verify it? Yes
- What issue was found? Footer PAGE field inserted in DOCX.
- How was it fixed? Page numbering present in rendered PDF.
- Final status: PASS

### Pass 24: Table of contents check
- What can go wrong here? TOC may be absent.
- Did I verify it? Yes
- What issue was found? Clean manually formatted Table of Contents included.
- How was it fixed? TOC present for supervisor review.
- Final status: PASS

### Pass 25: List of figures/list of tables check
- What can go wrong here? Lists may be absent or incomplete.
- Did I verify it? Yes
- What issue was found? List of Figures and List of Tables checked.
- How was it fixed? Both lists present and aligned with required captions.
- Final status: PASS

### Pass 26: Supervisor readability check
- What can go wrong here? The report may be too compressed for supervisor review.
- Did I verify it? Yes
- What issue was found? Expanded chapter explanations were included.
- How was it fixed? Supervisor-style narrative is fuller than the IEEE version.
- Final status: PASS

### Pass 27: Methodology defensibility check
- What can go wrong here? SMOTE placement or split design may be unclear.
- Did I verify it? Yes
- What issue was found? Stratified split, train/validation/test sizes, CV, preprocessing, and excluded features stated.
- How was it fixed? Methodology is defensible and reproducible.
- Final status: PASS

### Pass 28: Results interpretation check
- What can go wrong here? Results may be overstated.
- Did I verify it? Yes
- What issue was found? v3-v5 practical closeness and threshold-dependence emphasized.
- How was it fixed? Interpretation remains cautious.
- Final status: PASS

### Pass 29: Limitations honesty check
- What can go wrong here? Limitations may be minimized.
- Did I verify it? Yes
- What issue was found? Synthetic data, no external validation, proxy risk features, and missing uncertainty estimates stated.
- How was it fixed? Limitations retained.
- Final status: PASS

### Pass 30: Final send-readiness check
- What can go wrong here? Supervisor package may still be incomplete.
- Did I verify it? Yes
- What issue was found? PDF pages rendered: 33; structure clean: True
- How was it fixed? Ready for supervisor review after student-specific metadata is filled.
- Final status: PASS

## Final Verdict

**READY FOR SUPERVISOR REVIEW AFTER STUDENT-SPECIFIC METADATA IS FILLED.**
