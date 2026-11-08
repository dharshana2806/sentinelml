# Competitor analysis

Verification column: **V** = checked against a web source during this project (sources below); **U** = from general knowledge, *not verified* - check before quoting.

| Tool | Monitoring / drift | Performance (labels) | Label-free performance estimation | Feature-level diagnosis | Predictive (future) failure | Recovery workflows | Status |
|---|---|---|---|---|---|---|---|
| Evidently | V: 100+ metrics, drift tests, data quality, classification/regression reports, test suites for CI, monitoring dashboard | V | U | U | not seen in sources | not seen | V: open source, Evidently Cloud |
| NannyML | V | V | **V: CBPE (classification) & DLE (regression)** | V: G2 listing says it links drift alerts to features | not seen (it estimates *current* performance) | not seen | V: open source Apache-2.0 |
| WhyLabs | V: profiling (whylogs), monitoring | U | U | U | not seen | not seen | **V: hosted platform discontinued after Apple acquisition (Jan 2025); platform open-sourced** |
| Arize AI | U | U | U | U | U | U | not verified |
| Fiddler AI | U | U | U | U | U | U | not verified |

"Not seen" means *not found in the pages I read*, not "does not exist". Potential differentiation of SentinelML: (1) forecasting
onset over a horizon with lead-time metrics, (2) offline counterfactual ranking of recovery candidates with guards and mandatory human
approval. NannyML's label-free *estimation* of current performance is a complementary technique SentinelML does **not** implement
and could use as an additional signal (future work).

Sources: NannyML docs "Estimation of Performance of the Monitored Model" (nannyml.readthedocs.io/en/main/how_it_works/performance_estimation.html);
G2 NannyML listing; Evidently GitHub README (github.com/evidentlyai/evidently); ZenML docs and GeekWire (2025) on WhyLabs/Apple; aicoolies WhyLabs summary (Jan 23 2025 open-sourcing, hosted access ended Mar 9 2025).
