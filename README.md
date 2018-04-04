# BMI707 Project Proposal

## Team members:
Undina Gisladottir, Peter Shen, Jiaqi Xie

## Dataset:
Potentially using open claims data or MIMIC-III clinical notes

## Research Question:
Could we refer patients to the correct specializations for follow-ups based on their symptoms inferred from clinical notes?

## Deep Learning Approaches:
Natural Language Processing with RNNs such as LSTM, compare with state of art clinical note semantics neural networks

## Use case:
Patients are currently referred to specialists by primary physicians. Could we help primary physicians in referring patients to the correct specialists and eliminate administrative costs?

## Abstract:
Current ACO or HMO providers require a patient to visit their primary care physician and then be referred to the appropriate specialist after a series of workups. However, two separate Kyruuss surveys showed that almost 20 million referrals were “clinically inappropriate” and 90% of patients do not trust their physicians decision and will seek to verify the referrals themselves (Pennic 2014; Knowles 2017). In this study, our team will be using natural language processing (NLP) to predict clinically appropriate patient referrals to specializations based on patient symptoms inferred from clinical notes. To achieve this we will be gathering data from claims datasets and other open data sources and train our models using RNNs.

## References
Knowles, Megan. 2017. “Survey: 90% of Patients Seek to Validate Provider Referrals on Their Own.” December 5, 2017. https://www.beckershospitalreview.com/patient-engagement/survey-90-of-patients-seek-to-validate-provider-referrals-on-their-own.html.
Pennic, Jasmine. 2014. “19.7M ‘Clinically Inappropriate’ Physician Referrals Occur Each Year.” November 10, 2014. https://hitconsultant.net/2014/11/10/19-7m-clinically-inappropriate-physician-referrals-occur-each-year/.
