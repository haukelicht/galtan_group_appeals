In this step, we perform mention attribute dimension classification using the SetFit framework, specifically focusing on error detection.

1. We trained setfit classifiers to classify mentions into economic attributes and noneconomic attributes, respectively, using many different models and mention-in-context formatting strategies.
2. This yielded multiple classifciations per held-out mention-in-context instance from which we could compute misclassification rates (relative to the then current "true" labels).
3. We exported instances with high misclassification rates for manual review to identify potential annotation errors.

