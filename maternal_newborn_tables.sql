-- =====================================================================
-- MaaaLink (Find the Right Care): mother and newborn upgrade (version 5)
--
-- Changes from version 1:
--   * level 3 newborn care: no comma allowed between the neonatal word and
--     ventilation or surfactant
--   * "open 24 hours" alone no longer counts for the rating
--   * quotes no longer start with a comma or other punctuation
-- Change in version 3:
--   * a sentence is not counted when it says the service is not available, not
--     offered or not performed, or speaks of reducing or avoiding it
--   * facilities whose name points to eye, dental, skin, homeopathy or a
--     diagnostic centre are taken out of the mother and newborn lists (step 8b)
-- Change in version 4:
--   * five more signals for the questions page: newborn phototherapy, warmer or
--     incubator, paediatric intensive care, ambulance, maternal-fetal medicine.
--     They do not change who is listed or how a record is rated.
-- Change in version 5:
--   * seven more signals for planning ahead: normal delivery, pain relief in
--     labour (epidural), antenatal care, postnatal care, vaccination,
--     ultrasound, laboratory tests. Like the version 4 signals they are only
--     shown: they do not change who is listed or how a record is rated.
--   * the function also says whether the record lists paediatrics among its
--     specialties. A specialty tag is not evidence of a service; it is shown
--     as a hint when the record does not mention vaccination.
--
-- Run in the Databricks SQL editor, from top to bottom.
-- It is safe to run the whole file again: every step rebuilds its output.
-- Everything is created in workspace.care_finder. No source table is changed.
--
-- What it does
--   Step 0  keeps a copy of the two tables this file changes
--   Step 1  one row per sentence of every facility record
--   Step 2  finds the signals (C-section, blood, newborn care, ...) in those sentences
--   Step 3  service list: adds High-risk delivery
--   Step 4  evidence rows for High-risk delivery and for NICU
--   Step 5  24-hour wording about the facility
--   Step 6  24-hour wording next to a service
--   Step 7  one row per facility with its signals and its NICU level
--   Step 8  rating per facility and service
--   Step 9  the function the app calls
--   Step 10 checks (send the result back)
-- =====================================================================


-- ---------------------------------------------------------------------
-- STEP 0. Copies of the tables changed below (made the first time only)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS workspace.care_finder.facility_service_evidence_before_maternal AS
SELECT * FROM workspace.care_finder.facility_service_evidence;

CREATE TABLE IF NOT EXISTS workspace.care_finder.facility_service_summary_before_maternal AS
SELECT * FROM workspace.care_finder.facility_service_summary;


-- ---------------------------------------------------------------------
-- STEP 1. One row per sentence. A sentence repeated in several parts of
--         the same record is kept once.
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE workspace.care_finder.facility_sentences AS
WITH raw AS (
  SELECT unique_id, 'capability' AS field, explode(capability_arr) AS sentence
  FROM workspace.care_finder.facilities_final
  UNION ALL
  SELECT unique_id, 'procedure' AS field, explode(procedure_arr) AS sentence
  FROM workspace.care_finder.facilities_final
  UNION ALL
  SELECT unique_id, 'equipment' AS field, explode(equipment_arr) AS sentence
  FROM workspace.care_finder.facilities_final
  UNION ALL
  SELECT unique_id, 'description' AS field, description AS sentence
  FROM workspace.care_finder.facilities_final
  WHERE description IS NOT NULL
),
numbered AS (
  SELECT unique_id, field, sentence,
         row_number() OVER (
           PARTITION BY unique_id, lower(trim(sentence))
           ORDER BY CASE field WHEN 'capability' THEN 1 WHEN 'procedure' THEN 2
                               WHEN 'equipment' THEN 3 ELSE 4 END) AS rn
  FROM raw
  WHERE sentence IS NOT NULL AND trim(sentence) <> ''
)
SELECT unique_id, field, sentence, lower(sentence) AS s
FROM numbered
WHERE rn = 1;


-- ---------------------------------------------------------------------
-- STEP 2. Signal hits. One row per sentence and signal.
--   quote   = the clause of the sentence that contains the keyword
--   planned = the keyword sits next to planned, future or negated wording
--             ("not available", "does not offer", "reduce caesarean sections");
--             those rows are kept here for the record but never used.
-- The keyword patterns are drafts and have not been clinically reviewed.
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE workspace.care_finder.facility_signal_hits AS
WITH pat AS (
  SELECT * FROM VALUES
    ('csection',      '(^|[^a-z])c[- ]?section|ca?esar[ei]an|(^|[^a-z])lscs'),
    ('highrisk',      'high[- ]risk (pregnan|deliver|obstetric|matern)'),
    ('blood',         'blood bank|blood storage|blood transfusion|blood cent(re|er)'),
    ('surgery',       'ana?esthe(tist|sia|siolog)|operation theat|operating (room|theat)'),
    ('nicu',          '(^|[^a-z])nicu([^a-z]|$)|neonatal (intensive|icu)|newborn intensive'),
    ('nb_level3',     '(neonat|new ?born|(^|[^a-z])nicu([^a-z]|$)|(^|[^a-z])sncu([^a-z]|$))[^.;,]{0,60}(ventilat|surfactant)|(ventilat|surfactant)[^.;,]{0,60}(neonat|new ?born|(^|[^a-z])nicu([^a-z]|$)|(^|[^a-z])sncu([^a-z]|$))'),
    ('neonatologist', 'neonatologist'),
    ('nb_level2',     '(^|[^a-z])sncu([^a-z]|$)|special (new ?born|neonatal) care|(neonat|new ?born|(^|[^a-z])nicu([^a-z]|$)|(^|[^a-z])sncu([^a-z]|$))[^.;]{0,60}(cpap)|(cpap)[^.;]{0,60}(neonat|new ?born|(^|[^a-z])nicu([^a-z]|$)|(^|[^a-z])sncu([^a-z]|$))'),
    ('nb_level1',     '(^|[^a-z])nbsu([^a-z]|$)|new ?born stabili[sz]ation|radiant warmer|(neonat|new ?born|(^|[^a-z])nicu([^a-z]|$)|(^|[^a-z])sncu([^a-z]|$))[^.;]{0,60}(phototherapy)|(phototherapy)[^.;]{0,60}(neonat|new ?born|(^|[^a-z])nicu([^a-z]|$)|(^|[^a-z])sncu([^a-z]|$))'),
    ('newborn_any',   'neonat|(^|[^a-z])(nicu|sncu|nbsu)([^a-z]|$)|new ?born (intensive|stabili[sz]ation|care unit)|special new ?born care'),
    ('phototherapy',  'phototherap'),
    ('warmer',        'radiant warmer|baby warmer|infant warmer|(infant|neonatal|baby|new ?born|transport)[ ]incubator'),
    ('picu',          '(^|[^a-z])picu([^a-z]|$)|pa?ediatric (intensive|icu|critical)'),
    ('ambulance',     'ambulance'),
    ('mfm',           'maternal[- ]f(o)?etal|f(o)?etal medicine|perinatolog'),
    ('normal_delivery', '(^|[^a-z])normal[^.;,]{0,25}deliver|vaginal (birth|deliver)|natural (birth|deliver)|(^|[^a-z])nvd([^a-z]|$)'),
    ('painless',      'painless (deliver|labou?r|birth)|epidural|labou?r analgesia'),
    ('antenatal',     'ante[- ]?natal|pre[- ]?natal|(^|[^a-z])anc([^a-z]|$)|pregnancy (care|check)'),
    ('postnatal',     'post[- ]?natal|post[- ]?partum|(^|[^a-z])pnc([^a-z]|$)'),
    ('vaccination',   'vaccin|immuni[sz]ation'),
    ('ultrasound',    'ultrasound|ultrasonograph|sonograph|(^|[^a-z])usg([^a-z]|$)'),
    ('lab',           'laborator|patholog|blood test')
  AS t(sig, p)
),
m AS (
  SELECT fs.unique_id, pat.sig, pat.p, fs.field, fs.sentence, fs.s
  FROM workspace.care_finder.facility_sentences fs
  JOIN pat ON fs.s RLIKE pat.p
),
q AS (
  SELECT unique_id, sig, field, sentence,
         coalesce(
           nullif(regexp_replace(trim(regexp_extract(sentence,
                    concat('(?i)[^.;]*(', p, ')[^.;]*'), 0)), '^[,:) -]+', ''), ''),
           sentence) AS quote,
         (   s RLIKE concat('(plans? to (introduce|establish|start|set up|add|open|convert|build|launch)|planning to)[^.;]{0,120}(', p, ')')
          OR s RLIKE concat('(proposed|upcoming|coming soon|planned for|expanding with|without|does not have|do not have|does not (offer|provide|perform)|do not (offer|provide|perform)|not offering|no facilit(y|ies) for)[^.;]{0,40}(', p, ')')
          OR s RLIKE concat('(reduc(e|es|ing)|avoid(s|ing)?|instead of|alternative to)([ ](the|a|an|unnecessary|number of|rate of|need for))*[ ]?(', p, ')')
          OR s RLIKE concat('(', p, ')[^.;]{0,40}(planned|under construction|will be (installed|available|established|started|set up|added)|to be (installed|established|set up|started|added)|coming soon|without surgery)')
          OR s RLIKE concat('(', p, ')[^.;,]{0,25}(not (currently )?(available|offered|provided|performed)|unavailable)')
          OR (sig = 'phototherapy' AND s RLIKE 'psoria|vitiligo|dermat|skin|uvb|puva|eczema')) AS planned
  FROM m
)
SELECT unique_id, sig, field, sentence, quote, planned,
       CASE field WHEN 'capability' THEN 1 WHEN 'procedure' THEN 2
                  WHEN 'equipment' THEN 3 ELSE 4 END * 100000 + length(quote) AS w
FROM q;


-- ---------------------------------------------------------------------
-- STEP 3. Service list. High-risk delivery is new; the NICU keywords now
--         also cover lower levels of newborn care.
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE workspace.care_finder.service_taxonomy AS
SELECT * FROM VALUES
  ('dialysis', 'Dialysis',
   'dialys',
   'nephro|dialys|renal'),
  ('c_section', 'Emergency C-section',
   '(^|[^a-z])c[- ]?section|ca?esar[ei]an|(^|[^a-z])lscs',
   'obstetric|gyna?ecolog|maternal'),
  ('high_risk_delivery', 'High-risk delivery',
   '(^|[^a-z])c[- ]?section|ca?esar[ei]an|(^|[^a-z])lscs|high[- ]risk (pregnan|deliver|obstetric|matern)',
   'obstetric|gyna?ecolog|maternal'),
  ('nicu', 'NICU',
   '(^|[^a-z])nicu([^a-z]|$)|neonatal (intensive|icu)|newborn intensive|(neonat|new ?born|(^|[^a-z])nicu([^a-z]|$)|(^|[^a-z])sncu([^a-z]|$))[^.;,]{0,60}(ventilat|surfactant)|(ventilat|surfactant)[^.;,]{0,60}(neonat|new ?born|(^|[^a-z])nicu([^a-z]|$)|(^|[^a-z])sncu([^a-z]|$))|(^|[^a-z])sncu([^a-z]|$)|special (new ?born|neonatal) care|(neonat|new ?born|(^|[^a-z])nicu([^a-z]|$)|(^|[^a-z])sncu([^a-z]|$))[^.;]{0,60}(cpap)|(cpap)[^.;]{0,60}(neonat|new ?born|(^|[^a-z])nicu([^a-z]|$)|(^|[^a-z])sncu([^a-z]|$))|(^|[^a-z])nbsu([^a-z]|$)|new ?born stabili[sz]ation|radiant warmer|(neonat|new ?born|(^|[^a-z])nicu([^a-z]|$)|(^|[^a-z])sncu([^a-z]|$))[^.;]{0,60}(phototherapy)|(phototherapy)[^.;]{0,60}(neonat|new ?born|(^|[^a-z])nicu([^a-z]|$)|(^|[^a-z])sncu([^a-z]|$))',
   'neonat'),
  ('cardiac_surgery', 'Cardiac surgery',
   'cabg|coronary artery bypass|heart bypass|open[- ]heart|cardiac surg|heart surg|cardiothoracic|valve replacement',
   'cardiacsurg|cardiothoracic(surg|vascular|$)|cardiovascular(and)?thoracic|cardiovascularsurg|ctvs'),
  ('mri', 'MRI',
   '(^|[^a-z])mri([^a-z]|$)|magnetic resonance',
   'radiolog|radiodiagnos'),
  ('burn_care', 'Burn care',
   '(^|[^a-z])burns?([^a-z]|$)',
   'burn')
AS t(service_id, service_name, text_pattern, specialty_pattern);


-- ---------------------------------------------------------------------
-- STEP 4. Evidence rows for High-risk delivery and NICU, taken from the
--         signal hits. Rows of the other services are left as they are.
-- ---------------------------------------------------------------------
DELETE FROM workspace.care_finder.facility_service_evidence
WHERE service_id = 'high_risk_delivery'
   OR (service_id = 'nicu' AND field <> 'specialty');

INSERT INTO workspace.care_finder.facility_service_evidence
SELECT DISTINCT unique_id, 'high_risk_delivery' AS service_id, field, sentence
FROM workspace.care_finder.facility_signal_hits
WHERE NOT planned AND sig IN ('csection', 'highrisk');

INSERT INTO workspace.care_finder.facility_service_evidence
SELECT DISTINCT h.unique_id, 'nicu' AS service_id, h.field, h.sentence
FROM workspace.care_finder.facility_signal_hits h
WHERE NOT h.planned
  AND (h.sig IN ('nicu', 'nb_level3', 'nb_level2', 'nb_level1')
       OR (h.sig = 'neonatologist'
           AND h.unique_id IN (SELECT unique_id
                               FROM workspace.care_finder.facility_signal_hits
                               WHERE sig = 'nicu' AND NOT planned)));

INSERT INTO workspace.care_finder.facility_service_evidence
SELECT unique_id, 'high_risk_delivery' AS service_id, 'specialty' AS field, tag AS sentence
FROM (SELECT unique_id, explode(specialties_arr) AS tag
      FROM workspace.care_finder.facilities_final)
WHERE tag RLIKE 'obstetric|gyna?ecolog|maternal';


-- ---------------------------------------------------------------------
-- STEP 5. 24-hour wording about the facility as a whole.
--   emergency_24h: the 24-hour wording is next to emergency, casualty or trauma
--   open_24h:      the record says the facility is open or operates 24 hours
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE workspace.care_finder.facility_hours AS
WITH m AS (
  SELECT unique_id,
         regexp_replace(trim(regexp_extract(sentence,
              '(?i)[^.;]*((24[ ]?[x/*-][ ]?7|24[ -]?(hours|hour|hrs|hr)|round[ -]the[ -]clock|always open)[^.;]{0,30}(emergency|casualty|trauma)|(emergency|casualty|trauma)[^.;]{0,30}(24[ ]?[x/*-][ ]?7|24[ -]?(hours|hour|hrs|hr)|round[ -]the[ -]clock|always open))[^.;]*', 0)), '^[,:) -]+', '')  AS emg_clause,
         regexp_replace(trim(regexp_extract(sentence,
              '(?i)[^.;]*((open|opens|operates|operating|functions|functioning)[^.;]{0,12}(24[ ]?[x/*-][ ]?7|24[ -]?(hours|hour|hrs|hr)|round[ -]the[ -]clock|always open)|always open)[^.;]*', 0)), '^[,:) -]+', '') AS open_clause
  FROM workspace.care_finder.facility_sentences
  WHERE s RLIKE '24[ ]?[x/*-][ ]?7|24[ -]?(hours|hour|hrs|hr)|round[ -]the[ -]clock|always open'
),
c AS (
  SELECT unique_id,
         CASE WHEN emg_clause <> ''
               AND NOT lower(emg_clause) RLIKE 'suspended|closed|strike|withdrawal|plans? to|planned|upcoming|proposed|will be|to be |coming soon|under construction|(within|after|first|every|next)[ ](the[ ])?24' THEN emg_clause END AS emg,
         CASE WHEN open_clause <> ''
               AND NOT lower(open_clause) RLIKE 'suspended|closed|strike|withdrawal|plans? to|planned|upcoming|proposed|will be|to be |coming soon|under construction|(within|after|first|every|next)[ ](the[ ])?24' THEN open_clause END AS opn
  FROM m
)
SELECT unique_id,
       bool_or(emg IS NOT NULL)   AS emergency_24h,
       bool_or(opn IS NOT NULL)   AS open_24h,
       min_by(emg, length(emg))   AS emergency_quote,
       min_by(opn, length(opn))   AS open_quote
FROM c
WHERE emg IS NOT NULL OR opn IS NOT NULL
GROUP BY unique_id;


-- ---------------------------------------------------------------------
-- STEP 6. 24-hour wording in the same clause as the service itself.
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE workspace.care_finder.facility_service_24h AS
WITH c AS (
  SELECT e.unique_id, e.service_id,
         regexp_replace(trim(regexp_extract(e.sentence,
              concat('(?i)[^.;]*(', t.text_pattern, ')[^.;]*'), 0)), '^[,:) -]+', '') AS clause
  FROM workspace.care_finder.facility_service_evidence e
  JOIN workspace.care_finder.service_taxonomy t USING (service_id)
  WHERE e.field <> 'specialty'
)
SELECT unique_id, service_id, min_by(clause, length(clause)) AS quote
FROM c
WHERE lower(clause) RLIKE '24[ ]?[x/*-][ ]?7|24[ -]?(hours|hour|hrs|hr)|round[ -]the[ -]clock|always open'
  AND NOT lower(clause) RLIKE 'suspended|closed|strike|withdrawal|plans? to|planned|upcoming|proposed|will be|to be |coming soon|under construction|(within|after|first|every|next)[ ](the[ ])?24'
GROUP BY unique_id, service_id;


-- ---------------------------------------------------------------------
-- STEP 7. One row per facility: the quote found for each signal, and the
--         level of newborn care the record describes.
--   level3  ventilation or surfactant next to a neonatal word with no comma in
--           between ("NICU with ventilators" counts; "NICU, ICU, ventilator"
--           does not), or a NICU plus a neonatologist
--   level2  special newborn care unit (SNCU) or neonatal CPAP
--   unknown a NICU is named, with no equipment or staff detail
--   level1  stabilisation unit, radiant warmer or newborn phototherapy, no NICU
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE workspace.care_finder.facility_signals AS
WITH g AS (
  SELECT unique_id,
         min_by(quote, w) FILTER (WHERE sig = 'csection')      AS q_csection,
         min_by(quote, w) FILTER (WHERE sig = 'highrisk')      AS q_highrisk,
         min_by(quote, w) FILTER (WHERE sig = 'blood')         AS q_blood,
         min_by(quote, w) FILTER (WHERE sig = 'surgery')       AS q_surgery,
         min_by(quote, w) FILTER (WHERE sig = 'nicu')          AS q_nicu,
         min_by(quote, w) FILTER (WHERE sig = 'nb_level3')     AS q_l3,
         min_by(quote, w) FILTER (WHERE sig = 'neonatologist') AS q_neo,
         min_by(quote, w) FILTER (WHERE sig = 'nb_level2')     AS q_l2,
         min_by(quote, w) FILTER (WHERE sig = 'nb_level1')     AS q_l1,
         min_by(quote, w) FILTER (WHERE sig = 'newborn_any')   AS q_any,
         min_by(quote, w) FILTER (WHERE sig = 'phototherapy')  AS q_photo,
         min_by(quote, w) FILTER (WHERE sig = 'warmer')        AS q_warmer,
         min_by(quote, w) FILTER (WHERE sig = 'picu')          AS q_picu,
         min_by(quote, w) FILTER (WHERE sig = 'ambulance')     AS q_ambulance,
         min_by(quote, w) FILTER (WHERE sig = 'mfm')           AS q_mfm,
         min_by(quote, w) FILTER (WHERE sig = 'normal_delivery') AS q_normal,
         min_by(quote, w) FILTER (WHERE sig = 'painless')      AS q_painless,
         min_by(quote, w) FILTER (WHERE sig = 'antenatal')     AS q_antenatal,
         min_by(quote, w) FILTER (WHERE sig = 'postnatal')     AS q_postnatal,
         min_by(quote, w) FILTER (WHERE sig = 'vaccination')   AS q_vaccination,
         min_by(quote, w) FILTER (WHERE sig = 'ultrasound')    AS q_ultrasound,
         min_by(quote, w) FILTER (WHERE sig = 'lab')           AS q_lab
  FROM workspace.care_finder.facility_signal_hits
  WHERE NOT planned
  GROUP BY unique_id
)
SELECT unique_id,
       q_csection AS sig_csection,
       q_blood    AS sig_blood,
       coalesce(q_nicu, q_l3, q_l2, q_any) AS sig_newborn,
       q_surgery  AS sig_surgery,
       q_highrisk AS sig_highrisk,
       CASE WHEN q_l3 IS NOT NULL OR (q_nicu IS NOT NULL AND q_neo IS NOT NULL) THEN 'level3'
            WHEN q_l2 IS NOT NULL   THEN 'level2'
            WHEN q_nicu IS NOT NULL THEN 'unknown'
            WHEN q_l1 IS NOT NULL   THEN 'level1' END AS nicu_level,
       CASE WHEN q_l3 IS NOT NULL OR (q_nicu IS NOT NULL AND q_neo IS NOT NULL)
                                    THEN coalesce(q_l3, q_neo)
            WHEN q_l2 IS NOT NULL   THEN q_l2
            WHEN q_nicu IS NOT NULL THEN q_nicu
            WHEN q_l1 IS NOT NULL   THEN q_l1 END AS nicu_quote,
       q_l3        AS sig_ventilation,
       q_photo     AS sig_phototherapy,
       q_warmer    AS sig_warmer,
       q_picu      AS sig_picu,
       q_ambulance AS sig_ambulance,
       q_mfm       AS sig_mfm,
       q_normal      AS sig_normal_delivery,
       q_painless    AS sig_painless,
       q_antenatal   AS sig_antenatal,
       q_postnatal   AS sig_postnatal,
       q_vaccination AS sig_vaccination,
       q_ultrasound  AS sig_ultrasound,
       q_lab         AS sig_lab
FROM g;


-- ---------------------------------------------------------------------
-- STEP 8. Rating per facility and service.
--   "24 hours" below means: 24-hour emergency at the facility, or 24-hour
--   wording in the same clause as the service. "Open 24 hours" alone is
--   shown on the card but does not count for the rating.
--   Emergency C-section: C-section + blood + 24 hours = 1_strong;
--                        C-section + one of them = 2_medium; C-section only = 3_weak
--   High-risk delivery:  of C-section, blood, 24 hours, newborn care:
--                        4 = 1_strong; 3 = 2_medium; 2 or fewer = 3_weak
--   NICU:                level3 = 1_strong; level2 = 2_medium;
--                        unknown = 3_weak; level1 = 4_basic
--   Other services:      3 or more parts of the record = 1_strong; 2 = 2_medium; 1 = 3_weak
--   4_tag_only means only a specialty tag, no text; those are never recommended.
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE workspace.care_finder.facility_service_summary AS
WITH ev AS (
  SELECT unique_id, service_id, sentence, min(field) AS field
  FROM workspace.care_finder.facility_service_evidence
  WHERE field <> 'specialty'
  GROUP BY unique_id, service_id, sentence
),
txt AS (
  SELECT unique_id, service_id, COUNT(DISTINCT field) AS n_text_fields
  FROM ev
  GROUP BY unique_id, service_id
),
tag AS (
  SELECT DISTINCT unique_id, service_id
  FROM workspace.care_finder.facility_service_evidence
  WHERE field = 'specialty'
),
pairs AS (
  SELECT coalesce(txt.unique_id, tag.unique_id)   AS unique_id,
         coalesce(txt.service_id, tag.service_id) AS service_id,
         coalesce(txt.n_text_fields, 0)           AS n_text_fields,
         CASE WHEN tag.unique_id IS NOT NULL THEN 1 ELSE 0 END AS has_specialty_tag
  FROM txt
  FULL OUTER JOIN tag
    ON txt.unique_id = tag.unique_id AND txt.service_id = tag.service_id
),
sig AS (
  SELECT p.unique_id, p.service_id, p.n_text_fields, p.has_specialty_tag,
         CASE WHEN g.sig_csection IS NOT NULL THEN 1 ELSE 0 END AS has_a,
         CASE WHEN g.sig_blood    IS NOT NULL THEN 1 ELSE 0 END AS has_b,
         CASE WHEN h.unique_id IS NOT NULL
                OR coalesce(o.emergency_24h, false) THEN 1 ELSE 0 END AS has_c,
         CASE WHEN g.sig_newborn  IS NOT NULL THEN 1 ELSE 0 END AS has_d,
         g.nicu_level
  FROM pairs p
  LEFT JOIN workspace.care_finder.facility_signals g
    ON p.unique_id = g.unique_id
  LEFT JOIN workspace.care_finder.facility_hours o
    ON p.unique_id = o.unique_id
  LEFT JOIN workspace.care_finder.facility_service_24h h
    ON p.unique_id = h.unique_id AND p.service_id = h.service_id
)
SELECT unique_id, service_id, n_text_fields, has_specialty_tag,
       CASE
         WHEN n_text_fields = 0 THEN '4_tag_only'
         WHEN service_id = 'c_section' THEN
           CASE WHEN has_b + has_c = 2 THEN '1_strong'
                WHEN has_b + has_c = 1 THEN '2_medium'
                ELSE '3_weak' END
         WHEN service_id = 'high_risk_delivery' THEN
           CASE WHEN has_a + has_b + has_c + has_d = 4 THEN '1_strong'
                WHEN has_a + has_b + has_c + has_d = 3 THEN '2_medium'
                ELSE '3_weak' END
         WHEN service_id = 'nicu' THEN
           CASE nicu_level WHEN 'level3' THEN '1_strong'
                           WHEN 'level2' THEN '2_medium'
                           WHEN 'level1' THEN '4_basic'
                           ELSE '3_weak' END
         WHEN n_text_fields >= 3 THEN '1_strong'
         WHEN n_text_fields = 2  THEN '2_medium'
         ELSE '3_weak'
       END AS evidence_tier
FROM sig;


-- ---------------------------------------------------------------------
-- STEP 8b. Take out facilities whose name points to a specialty with no
--          birth or newborn care: eye, dental, skin, homeopathy, and
--          diagnostic centres that are not hospitals. Their records carry
--          sentences such as "10-bed NICU" that most likely belong to
--          another facility. A name that also says maternity, women,
--          child, general, multi-speciality and the like is kept.
--          The list of what was taken out stays in its own table.
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE workspace.care_finder.facility_excluded_by_name AS
SELECT unique_id, name,
       CASE WHEN lower(name) RLIKE '(^|[^a-z])eye([^a-z]|$)|eyetec|(^|[^a-z])netra|ophthal'    THEN 'eye'
            WHEN lower(name) RLIKE 'dental|dentist' THEN 'dental'
            WHEN lower(name) RLIKE 'homeo'  THEN 'homeopathy'
            WHEN lower(name) RLIKE '(^|[^a-z])skin([^a-z]|$)|dermat'   THEN 'skin'
            ELSE 'diagnostic centre' END AS reason
FROM workspace.care_finder.facilities_final
WHERE NOT lower(name) RLIKE 'matern|wom[ae]n|mother|child|pa?ediatr|neonat|gyn|obstet|fertil|(^|[^a-z])ivf([^a-z]|$)|multi|general|super.?special|medical college|medical sciences|nursing home'
  AND (   lower(name) RLIKE '(^|[^a-z])eye([^a-z]|$)|eyetec|(^|[^a-z])netra|ophthal'
       OR lower(name) RLIKE 'dental|dentist'
       OR lower(name) RLIKE 'homeo'
       OR lower(name) RLIKE '(^|[^a-z])skin([^a-z]|$)|dermat'
       OR (lower(name) RLIKE '(^|[^a-z])scan([^a-z]|$)|diagnos|patholog|laborator|imaging' AND NOT lower(name) RLIKE 'hospital'));

DELETE FROM workspace.care_finder.facility_service_summary
WHERE service_id IN ('c_section', 'high_risk_delivery', 'nicu')
  AND unique_id IN (SELECT unique_id FROM workspace.care_finder.facility_excluded_by_name);


-- ---------------------------------------------------------------------
-- STEP 9. The function the app calls. Same inputs as before; it now also
--         returns the 24-hour information and the signals.
-- ---------------------------------------------------------------------
CREATE OR REPLACE FUNCTION workspace.care_finder.recommend_at(
  p_service   STRING,
  p_lat       DOUBLE,
  p_lon       DOUBLE,
  p_radius_km DOUBLE DEFAULT 100,
  p_sort      STRING DEFAULT 'evidence',
  p_limit     INT    DEFAULT 15)
RETURNS TABLE (
  result_rank           INT,
  list_section          STRING,
  unique_id             STRING,
  name                  STRING,
  address_line1         STRING,
  city                  STRING,
  state                 STRING,
  pin                   STRING,
  phone                 STRING,
  website               STRING,
  n_phone_numbers       INT,
  evidence_tier         STRING,
  distance_km           DOUBLE,
  distance_km_by_coords DOUBLE,
  location_quality      STRING,
  lat                   DOUBLE,
  lon                   DOUBLE,
  n_sources             INT,
  hours_level           STRING,
  hours_quote           STRING,
  sig_csection          STRING,
  sig_blood             STRING,
  sig_newborn           STRING,
  sig_surgery           STRING,
  sig_highrisk          STRING,
  nicu_level            STRING,
  nicu_quote            STRING,
  evidence_quotes       ARRAY<STRING>,
  source_urls           ARRAY<STRING>,
  sig_ventilation       STRING,
  sig_phototherapy      STRING,
  sig_warmer            STRING,
  sig_picu              STRING,
  sig_ambulance         STRING,
  sig_mfm               STRING,
  sig_normal_delivery   STRING,
  sig_painless          STRING,
  sig_antenatal         STRING,
  sig_postnatal         STRING,
  sig_vaccination       STRING,
  sig_ultrasound        STRING,
  sig_lab               STRING,
  specialty_paediatrics BOOLEAN)
RETURN
WITH once AS (
  SELECT unique_id, sentence, min(field) AS field
  FROM workspace.care_finder.facility_service_evidence
  WHERE service_id = p_service AND field <> 'specialty'
  GROUP BY unique_id, sentence
),
quotes AS (
  SELECT unique_id, collect_list(concat(field, ': ', sentence)) AS q
  FROM once
  GROUP BY unique_id
),
cand AS (
  SELECT f.unique_id, f.name, c.address_line1, f.address_city, f.address_stateOrRegion,
         f.pin_raw,
         CASE WHEN c.phone RLIKE '^[+]91[0-9]{8,12}$' THEN c.phone END AS phone,
         c.website, c.n_phone_numbers,
         s.evidence_tier, f.location_quality,
         CASE WHEN f.location_quality = 'conflict_unresolved'
              THEN '2_location_uncertain' ELSE '1_main' END AS list_section,
         workspace.care_finder.km(p_lat, p_lon, f.lat_final, f.lon_final) AS km_used,
         CASE WHEN f.location_quality = 'conflict_unresolved'
              THEN workspace.care_finder.km(p_lat, p_lon, f.lat_raw, f.lon_raw) END AS km_by_coords,
         f.lat_final, f.lon_final,
         size(f.source_urls_arr) AS n_sources,
         CASE WHEN h.unique_id IS NOT NULL       THEN 'service_24h'
              WHEN coalesce(o.emergency_24h, false) THEN 'emergency_24h'
              WHEN coalesce(o.open_24h, false)      THEN 'open_24h'
              ELSE 'not_stated' END AS hours_level,
         CASE WHEN h.unique_id IS NOT NULL       THEN h.quote
              WHEN coalesce(o.emergency_24h, false) THEN o.emergency_quote
              WHEN coalesce(o.open_24h, false)      THEN o.open_quote END AS hours_quote,
         g.sig_csection, g.sig_blood, g.sig_newborn, g.sig_surgery, g.sig_highrisk,
         g.nicu_level, g.nicu_quote,
         quotes.q, f.source_urls_arr,
         g.sig_ventilation, g.sig_phototherapy, g.sig_warmer, g.sig_picu, g.sig_ambulance,
         g.sig_mfm,
         g.sig_normal_delivery, g.sig_painless, g.sig_antenatal, g.sig_postnatal,
         g.sig_vaccination, g.sig_ultrasound, g.sig_lab,
         exists(coalesce(f.specialties_arr, array()), x -> lower(x) RLIKE 'pa?ediatr') AS specialty_paediatrics
  FROM workspace.care_finder.facility_service_summary s
  JOIN workspace.care_finder.facilities_final f
    ON s.unique_id = f.unique_id
  JOIN quotes
    ON s.unique_id = quotes.unique_id
  LEFT JOIN workspace.care_finder.facility_contact c
    ON s.unique_id = c.unique_id
  LEFT JOIN workspace.care_finder.facility_signals g
    ON s.unique_id = g.unique_id
  LEFT JOIN workspace.care_finder.facility_hours o
    ON s.unique_id = o.unique_id
  LEFT JOIN workspace.care_finder.facility_service_24h h
    ON s.unique_id = h.unique_id AND h.service_id = p_service
  WHERE s.service_id = p_service
    AND s.evidence_tier <> '4_tag_only'
    AND f.lat_final IS NOT NULL
),
ranked AS (
  SELECT *, row_number() OVER (
           ORDER BY list_section,
                    CASE WHEN p_sort = 'distance' THEN km_used END,
                    evidence_tier, km_used) AS rn
  FROM cand
  WHERE least(km_used, coalesce(km_by_coords, km_used)) <= p_radius_km
)
SELECT CAST(rn AS INT), list_section, unique_id, name, address_line1,
       address_city, address_stateOrRegion, pin_raw, phone, website, n_phone_numbers,
       evidence_tier, km_used, km_by_coords, location_quality,
       lat_final, lon_final, n_sources,
       hours_level, hours_quote,
       sig_csection, sig_blood, sig_newborn, sig_surgery, sig_highrisk,
       nicu_level, nicu_quote,
       q, source_urls_arr,
       sig_ventilation, sig_phototherapy, sig_warmer, sig_picu, sig_ambulance, sig_mfm,
       sig_normal_delivery, sig_painless, sig_antenatal, sig_postnatal,
       sig_vaccination, sig_ultrasound, sig_lab,
       specialty_paediatrics
FROM ranked
WHERE rn <= p_limit;


-- ---------------------------------------------------------------------
-- STEP 10. Checks. Run this last and send the result back.
-- ---------------------------------------------------------------------
SELECT '1 rating after'  AS check_name, service_id AS item, evidence_tier AS detail, COUNT(*) AS n
FROM workspace.care_finder.facility_service_summary
WHERE evidence_tier <> '4_tag_only'
GROUP BY service_id, evidence_tier
UNION ALL
SELECT '2 rating before', service_id, evidence_tier, COUNT(*)
FROM workspace.care_finder.facility_service_summary_before_maternal
WHERE evidence_tier <> '4_tag_only'
GROUP BY service_id, evidence_tier
UNION ALL
SELECT '3 newborn level', coalesce(nicu_level, 'none'), '', COUNT(*)
FROM workspace.care_finder.facility_signals
GROUP BY coalesce(nicu_level, 'none')
UNION ALL
SELECT '4 signal hits', sig, CASE WHEN planned THEN 'planned, not used' ELSE 'used' END,
       COUNT(DISTINCT unique_id)
FROM workspace.care_finder.facility_signal_hits
GROUP BY sig, CASE WHEN planned THEN 'planned, not used' ELSE 'used' END
UNION ALL
SELECT '5 24-hour level', 'emergency_24h', '', COUNT(*)
FROM workspace.care_finder.facility_hours WHERE emergency_24h
UNION ALL
SELECT '5 24-hour level', 'open_24h', '', COUNT(*)
FROM workspace.care_finder.facility_hours WHERE open_24h
UNION ALL
SELECT '5 24-hour level', 'service_24h', service_id, COUNT(*)
FROM workspace.care_finder.facility_service_24h
GROUP BY service_id
UNION ALL
SELECT '5 24-hour level', 'open_24h only, no longer counted', s.service_id, COUNT(*)
FROM workspace.care_finder.facility_service_summary s
JOIN workspace.care_finder.facility_hours o ON s.unique_id = o.unique_id
LEFT JOIN workspace.care_finder.facility_service_24h h
  ON s.unique_id = h.unique_id AND s.service_id = h.service_id
WHERE s.service_id IN ('c_section', 'high_risk_delivery')
  AND s.evidence_tier <> '4_tag_only'
  AND o.open_24h AND NOT o.emergency_24h AND h.unique_id IS NULL
GROUP BY s.service_id
UNION ALL
SELECT '7 for information', 'old C-section list (not shown in the app): rows whose quote is now excluded', '', COUNT(*)
FROM workspace.care_finder.facility_service_summary s
LEFT JOIN workspace.care_finder.facility_signals g ON s.unique_id = g.unique_id
WHERE s.service_id = 'c_section' AND s.evidence_tier <> '4_tag_only'
  AND g.sig_csection IS NULL
UNION ALL
SELECT '6 must be 0', 'mother list: facility with no usable C-section or high-risk sentence', '', COUNT(*)
FROM workspace.care_finder.facility_service_summary s
LEFT JOIN workspace.care_finder.facility_signals g ON s.unique_id = g.unique_id
WHERE s.service_id = 'high_risk_delivery' AND s.evidence_tier <> '4_tag_only'
  AND g.sig_csection IS NULL AND g.sig_highrisk IS NULL
UNION ALL
SELECT '6 must be 0', 'NICU listed without a newborn level', '', COUNT(*)
FROM workspace.care_finder.facility_service_summary s
LEFT JOIN workspace.care_finder.facility_signals g ON s.unique_id = g.unique_id
WHERE s.service_id = 'nicu' AND s.evidence_tier <> '4_tag_only'
  AND g.nicu_level IS NULL
UNION ALL
SELECT '8 taken out by name', x.reason, e.service_id, COUNT(DISTINCT e.unique_id)
FROM workspace.care_finder.facility_excluded_by_name x
JOIN workspace.care_finder.facility_service_evidence e ON x.unique_id = e.unique_id
WHERE e.service_id IN ('high_risk_delivery', 'nicu') AND e.field <> 'specialty'
GROUP BY x.reason, e.service_id
UNION ALL
SELECT '9 distinct facilities', 'mother or newborn list', '', COUNT(DISTINCT unique_id)
FROM workspace.care_finder.facility_service_summary
WHERE service_id IN ('high_risk_delivery', 'nicu') AND evidence_tier <> '4_tag_only'
UNION ALL
SELECT 'A extra signals on the lists', item, service_id, n FROM (
  SELECT s.service_id, stack(13,
           'ventilation',   count_if(g.sig_ventilation  IS NOT NULL),
           'phototherapy',  count_if(g.sig_phototherapy IS NOT NULL),
           'warmer',        count_if(g.sig_warmer       IS NOT NULL),
           'picu',          count_if(g.sig_picu         IS NOT NULL),
           'ambulance',     count_if(g.sig_ambulance    IS NOT NULL),
           'mfm',           count_if(g.sig_mfm          IS NOT NULL),
           'normal_delivery', count_if(g.sig_normal_delivery IS NOT NULL),
           'painless',      count_if(g.sig_painless     IS NOT NULL),
           'antenatal',     count_if(g.sig_antenatal    IS NOT NULL),
           'postnatal',     count_if(g.sig_postnatal    IS NOT NULL),
           'vaccination',   count_if(g.sig_vaccination  IS NOT NULL),
           'ultrasound',    count_if(g.sig_ultrasound   IS NOT NULL),
           'lab',           count_if(g.sig_lab          IS NOT NULL)) AS (item, n)
  FROM workspace.care_finder.facility_service_summary s
  JOIN workspace.care_finder.facility_signals g ON s.unique_id = g.unique_id
  WHERE s.service_id IN ('high_risk_delivery', 'nicu') AND s.evidence_tier <> '4_tag_only'
  GROUP BY s.service_id)
UNION ALL
SELECT 'B paediatrics listed as a specialty', 'on the list', s.service_id, COUNT(*)
FROM workspace.care_finder.facility_service_summary s
JOIN workspace.care_finder.facilities_final f ON s.unique_id = f.unique_id
WHERE s.service_id IN ('high_risk_delivery', 'nicu') AND s.evidence_tier <> '4_tag_only'
  AND exists(coalesce(f.specialties_arr, array()), x -> lower(x) RLIKE 'pa?ediatr')
GROUP BY s.service_id
ORDER BY check_name, item, detail;
