Process for generating datasets and models:
1. Run Synthetic Data Generation notebook. This would create the underlying synthetic datasets in input
2. Run "python query_generation.py" script. To create workloads with past region evaluations
3. Run "python model_training.py" to train models on queries

Accuracy Experiments:
1. Run the Accuracy-Synthetic notebook - This would generate the first graphs 
for the Accuracy Experiments
Qualitative Experiments
1. Run Crimes-Qualitative
2. Run Human-Activity-Qualitative

Performance 
1. Run Performance notebook
2. Run training overhead python script

Sensitivty Experiments

1. For GlowWorm : Run the GlowWorm-Sensitivty notebook
2. For Model Sensitivity : Run Testing ML Algos
