# MaaaLink

Find where to take a mother or newborn, and what to ask before you go.

CMU Data Science Club Datathon, Fall 2026. Challenge: "Find the right care" (Virtue Foundation dataset of healthcare facilities in India, on Databricks Free Edition).

## What it does

MaaaLink is a Databricks App (Streamlit). For a place in India it shows which facilities have a record that mentions the maternal or newborn care needed. For each facility it quotes the wording from the record, says what the record leaves out, and marks records that look doubtful.

- **Urgent path.** Choose the situation, tick what is happening, enter a place. The app suggests a first choice and a backup, gives a "Call now" button, and lists what to ask on the phone.
- **Plan ahead.** The full list for an area, a coverage map, filters, and a one-page summary to print. When planning a birth you can say what you would like (normal delivery, pain relief in labour, checkups, care after the birth, vaccination, tests); each box then shows which of these the record mentions, and the list can be ordered by best match.
- **Show to driver.** A large-print card with the address and two sentences in the main language of the facility's state.
- **Ask Genie.** Free questions about the data, answered by a Databricks Genie agent over the same tables.
- **Method page.** A short version for everyone and a technical version with the decision model as formulas.

## The principle

The tool does not promise that a suggestion is right. It shows what each record says and how far that can be trusted, and says so plainly when the evidence is weak.

## Files

| File | What it is |
|---|---|
| `MaaaLink_datathon.ipynb` | Everything in one notebook: the SQL and the app files |
| `maternal_newborn_tables.sql` | Builds the mother and newborn tables and the function the app calls (version 5) |
| `app.py`, `app.yaml`, `requirements.txt` | The Databricks App |

The SQL builds on tables made earlier in the same schema (`workspace.care_finder`): the cleaned facility table `facilities_final`, the contact table `facility_contact`, and the place and PIN lookups. Those steps are not in this repository.

## How to run

1. Run `maternal_newborn_tables.sql` in the Databricks SQL editor, top to bottom.
2. Create a Databricks App from `app.py`, `app.yaml` and `requirements.txt`.
3. Add two app resources: a SQL warehouse with key `sql-warehouse`, and a Genie agent with key `genie-space`.

## Numbers from the real data

- Mother list: 660 facilities (23 mention all 4 items, 108 mention 3 of 4, 529 mention 2 or fewer).
- Newborn list: 757 facilities (126 NICU with ventilator or specialist, 22 special newborn unit or CPAP, 592 NICU named with no details, 17 basic newborn care only).
- 1,241 distinct facilities across both lists.

## Limits

- Every statement in a record is a claim taken from web pages. Nobody has confirmed it.
- Distances are straight lines.
- The decision model's stand-in for "how likely a facility is to help" and its 600 km penalty are our assumptions.
- Which situation or sign looks for what was decided by the team and has not been checked against a clinical source.
- The driver-card sentences in nine Indian languages have not been checked by native speakers.

## Next steps

- **A view by district.** For each district: how many facilities on our lists, next to three NFHS-5 indicators (institutional births, births by C-section, four or more antenatal visits). A teammate's data pipeline already has a first version of this idea. A first check with our own lists: NFHS-5 has 706 districts; 303 districts have at least one facility on our lists; only 206 of those match an NFHS-5 district name as written, so the names must be reconciled before anything is shown. In such a view a zero means "no record in this dataset", not "no care".
- **Stronger checks for records that mix facilities**, for example sentences that name another facility in the dataset, or several different cities in one record.
- **Review** of the links between a situation or sign and what the tool looks for by a clinician, and of the driver-card sentences by native speakers.

## References

- WHO, *Monitoring emergency obstetric care: a handbook* (2009). Comprehensive emergency obstetric care: the basic functions plus caesarean section and blood transfusion. https://www.who.int/publications-detail-redirect/9789241547734
- First Referral Units in India: caesarean section, newborn care and blood storage, available 24 hours. https://pmc.ncbi.nlm.nih.gov/articles/5144115
- Ministry of Health and Family Welfare, Government of India, on PMSMA: women with a high-risk pregnancy are linked to the nearest First Referral Unit. https://mohfw.gov.in/press-info/7478
- National Health Mission, *Facility Based Newborn Care Operational Guidelines 2025*: newborn stabilisation units and special newborn care units. https://nhm.gov.in/images/pdf/programmes/CH-Programmes/FBNC/FBNC%20Operational%20Guidelines%202025.pdf
- National Neonatology Forum of India, accreditation form for level IIIB units (2022-23): levels of newborn care by type of breathing support. https://www.nnfi.org/assests/upload/download-pdf/FINAL_FORM_LEVEL_IIIB_2022-23.pdf
- Emergency numbers 112, 102 and 108: the india.gov.in helpline directory and a Ministry of Health and Family Welfare reply in Parliament (no link kept).
- Databricks Free Edition limitations. https://docs.databricks.com/aws/en/getting-started/free-edition-limitations
- Data: the Virtue Foundation dataset of healthcare facilities in India, provided for the challenge. Challenge brief: https://docs.google.com/document/d/e/2PACX-1vQxhDNQmgZhPgjG0M6LCb_UXNV8CJryqBNQQuqWfIg_dm40ZU1AYAXzZKViC6haFoCmLNiRVfUSBMjf/pub

We worked from summaries of the guideline documents, not the full texts.
