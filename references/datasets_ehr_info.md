# EHR Datasets

| Dataset | Best suited for | Access | Notes |
|---------|-----------------|--------|-------|
| MIMIC-IV | Mortality, readmission, length-of-stay, diagnosis and ICU outcome classification | Credentialed public access | Large de-identified EHR database from Beth Israel Deaconess Medical Center; requires PhysioNet registration, training, and a data-use agreement. |
| eICU Collaborative Research Database | ICU mortality, deterioration, diagnosis, and treatment-outcome prediction | Credentialed public access | Multi-hospital ICU dataset, useful when you need broader institutional diversity than MIMIC-IV. |
| PhysioNet Challenge datasets | Sepsis, mortality, ECG classification, clinical deterioration | Usually open or credentialed, depending on year | Good choice for well-defined classification tasks with standardized evaluation scripts. |
| HiRID | Early clinical deterioration, mortality, and ICU event prediction | Credentialed access | High-resolution ICU data with frequent vital-sign measurements; useful for temporal classification. |
| Synthea | Prototyping, pipeline development, FHIR-based classification | Fully open synthetic data | No real-patient privacy risk, but synthetic data may not reflect real-world clinical distributions. |
| OMOP-compatible synthetic EHR datasets | Testing cohort definitions, feature engineering, and reproducible clinical ML pipelines | Generally open | Useful for developing code before applying it to restricted clinical data. |
