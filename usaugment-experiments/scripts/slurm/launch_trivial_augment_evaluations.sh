#!/bin/bash

sbatch --time=21:00:00 --job-name=aul_liver_segmentation usaugment-experiments/train_trivial_augment.sh aul_liver_v5 aul_liver_segmentation
sbatch --time=12:00:00 --job-name=aul_mass_classification usaugment-experiments/train_trivial_augment.sh aul_mass_v5 aul_mass_classification
sbatch --time=12:00:00 --job-name=aul_mass_segmentation usaugment-experiments/train_trivial_augment.sh aul_mass_v5 aul_mass_segmentation

sbatch --time=135:00:00 --job-name=butterfly_classification usaugment-experiments/train_trivial_augment.sh butterfly_v4 butterfly_classification

sbatch --time=36:00:00 --job-name=camus_classification usaugment-experiments/train_trivial_augment.sh camus_v4 camus_classification
sbatch --time=123:00:00 --job-name=camus_segmentation usaugment-experiments/train_trivial_augment.sh camus_v4 camus_segmentation

sbatch --time=12:00:00 --job-name=fatty_liver_classification usaugment-experiments/train_trivial_augment.sh fatty_liver_v3 fatty_liver_classification

sbatch --time=24:00:00 --job-name=gbcu_classification usaugment-experiments/train_trivial_augment.sh gbcu_v2 gbcu_classification

sbatch --time=21:00:00 --job-name=mmotu_classification usaugment-experiments/train_trivial_augment.sh mmotu_v2 mmotu_classification
sbatch --time=36:00:00 --job-name=mmotu_segmentation usaugment-experiments/train_trivial_augment.sh mmotu_v2 mmotu_segmentation

sbatch --time=24:00:00 --job-name=open_kidney_capsule_segmentation usaugment-experiments/train_trivial_augment.sh open_kidney_v3 open_kidney_capsule_segmentation

sbatch --time=24:00:00 --job-name=pocus_classification usaugment-experiments/train_trivial_augment.sh pocus_v4 pocus_classification

sbatch --time=33:00:00 --job-name=psfhs_segmentation usaugment-experiments/train_trivial_augment.sh psfhs_v2 psfhs_segmentation

sbatch --time=87:00:00 --job-name=stanford_thyroid_segmentation usaugment-experiments/train_trivial_augment.sh stanford_thyroid_v4 stanford_thyroid_segmentation
