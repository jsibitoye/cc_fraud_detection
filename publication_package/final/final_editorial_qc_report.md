# Final Editorial QC Report

Date: 2026-06-17

## 1. Files Generated

- `publication_package/final/manuscript_ieee_final.docx`
- `publication_package/final/manuscript_ieee_final.pdf`
- `publication_package/final/final_editorial_qc_report.md`
- `publication_package/final/final_claims_audit.md`
- `publication_package/final/final_reference_audit.md`
- `publication_package/final/final_formatting_audit.md`
- `publication_package/final/final_submission_checklist.md`

## 2. Files Copied

- Figures copied to `publication_package/final/figures/`
- Tables copied to `publication_package/final/tables/`
- References copied to `publication_package/final/references/`

## 3. Sections Revised

- Abstract, Introduction, Related Work, Dataset Construction and Validation, Methodology, Results, Discussion, Threats to Validity, Data/Ethics/Funding/Competing Interests, Conclusion, and References.

## 4. Tables Fixed

- All seven tables are real Word tables.
- Table VI column widths were adjusted to prevent a broken Version column.

## 5. Figures Fixed

- All ten final figures were copied, embedded, centered, captioned, and visually checked in the rendered PDF.

## 6. References Audited

- See `final_reference_audit.md`.

## 7. Claims Audited

- See `final_claims_audit.md`.

## 8. Formatting Audited

- See `final_formatting_audit.md`.

## 9. Forbidden-String Scan Result

- Status: PASS
- Hits: `{}`

## 10. Remaining Weaknesses

- Synthetic dataset only.
- No external validation dataset.
- Proxy risk indicators require feature-provenance caution.
- No repeated-seed experiments, confidence intervals, bootstrap intervals, or formal significance tests.
- Final venue template and author metadata are still required.

## 11. Human Actions Still Required

- Add real author names, affiliations, corresponding-author marker, and ORCID values if required.
- Confirm the target venue and apply its official IEEE template rules.
- Verify reference metadata and DOI completeness manually.
- Decide whether to add repeated-seed/statistical robustness experiments before submission.

## 12. Final Verdict

**NEAR READY BUT NEEDS STATISTICAL ROBUSTNESS**

## Automated Checks

| Check | Status | Detail |
|---|---|---|
| manuscript_ieee_final.docx exists | PASS | C:\Dev\cc_fraud_detection\publication_package\final\manuscript_ieee_final.docx |
| manuscript_ieee_final.pdf exists | PASS | C:\Dev\cc_fraud_detection\publication_package\final\manuscript_ieee_final.pdf |
| PDF rendered to page images | PASS | 14 pages |
| abstract under 250 words | PASS | 183 words |
| keyword count is 6 to 8 | PASS | 8 keywords |
| all 10 figures exist | PASS | 10/10 |
| all 10 figures embedded | PASS | 10/10 |
| all 10 figure captions present | PASS | 10/10 |
| seven Word tables present | PASS | 7 tables |
| seven table captions present | PASS | 7/7 |
| forbidden-string scan | PASS | {} |
| references present | PASS | 48 bracketed citations/references |
| required final audit files exist | PASS | audit/checklist files |
