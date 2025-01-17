#!/bin/bash

# This file is an updated version of the file `get_and_process_web_data.sh` from
# the official repository (https://github.com/jannisborn/covid19_ultrasound/).
# We kep only the links for convex images and removed links that are not working
# or link to images that contain more than one scan. Please refer to the
# original repository for more details.

# Default output directory
output_dir="tmp"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -o|--output) output_dir="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

mkdir -p "$output_dir/pocus_videos/convex"
mkdir -p "$output_dir/pocus_images/convex"

echo "Downloading files to $output_dir..."

wget -q -O $output_dir/pocus_videos/convex/Cov-clarius.gif https://clarius.com/wp-content/uploads/2020/03/1-blines.gif && echo "Downloaded Cov-clarius.gif" || echo "Failed to download Cov-clarius.gif"
wget -q -O $output_dir/pocus_videos/convex/Cov-clarius3.gif https://clarius.com/wp-content/uploads/2020/03/3-large-consolidation.gif && echo "Downloaded Cov-clarius3.gif" || echo "Failed to download Cov-clarius3.gif"
wget -q -O $output_dir/pocus_videos/convex/pneu-everyday.gif https://bit.ly/39wgXfk && echo "Downloaded pneu-everyday.gif" || echo "Failed to download pneu-everyday.gif"
wget -q -O $output_dir/pocus_videos/convex/Reg-nephropocus.gif https://nephropocushome.files.wordpress.com/2019/07/a-lines-titled.gif && echo "Downloaded Reg-nephropocus.gif" || echo "Failed to download Reg-nephropocus.gif"
wget -q -O $output_dir/pocus_videos/convex/Reg-bcpocus.gif https://www.bcpocus.ca/wp-content/uploads/2018/07/A-lines.gif && echo "Downloaded Reg-bcpocus.gif" || echo "Failed to download Reg-bcpocus.gif"
wget -q -O $output_dir/pocus_videos/convex/Cov-grepmed-blines-pocus.mp4 https://img.grepmed.com/uploads/7415/blines-pocus-pulmonary-lung-covid19-original.mp4 && echo "Downloaded Cov-grepmed-blines-pocus.mp4" || echo "Failed to download Cov-grepmed-blines-pocus.mp4"
wget -q -O $output_dir/pocus_videos/convex/Cov-grepmed2.mp4 https://img.grepmed.com/uploads/7410/sparing-lung-ultrasound-subpleural-blines-original.mp4 && echo "Downloaded Cov-grepmed2.mp4" || echo "Failed to download Cov-grepmed2.mp4"
wget -q -O $output_dir/pocus_videos/convex/pneu-gred-6.gif https://img.grepmed.com/uploads/5721/airbronchogram-pulmonary-pocus-pneumonia-lung-original.gif && echo "Downloaded pneu-gred-6.gif" || echo "Failed to download pneu-gred-6.gif"
wget -q -O $output_dir/pocus_videos/convex/Pneu-grep-pneumonia1.mp4 https://img.grepmed.com/uploads/6903/ultrasound-pocus-bronchograms-lung-pulmonary-original.mp4 && echo "Downloaded Pneu-grep-pneumonia1.mp4" || echo "Failed to download Pneu-grep-pneumonia1.mp4"
wget -q -O $output_dir/pocus_videos/convex/Pneu-grep-pneumonia2_1.mp4 https://img.grepmed.com/uploads/6876/bronchograms-air-pulmonary-pocus-ultrasound-original.mp4 && echo "Downloaded Pneu-grep-pneumonia2_1.mp4" || echo "Failed to download Pneu-grep-pneumonia2_1.mp4"
wget -q -O $output_dir/pocus_videos/convex/Pneu-grep-pneumonia4.mp4 https://img.grepmed.com/uploads/6431/ultrasound-lung-pneumonia-shredsign-clinical-original.mp4 && echo "Downloaded Pneu-grep-pneumonia4.mp4" || echo "Failed to download Pneu-grep-pneumonia4.mp4"
wget -q -O $output_dir/pocus_videos/convex/pneu-gred-7.gif https://img.grepmed.com/uploads/1304/airbronchograms-pneumonia-sonostuff-effusion-clinical-original.gif && echo "Downloaded pneu-gred-7.gif" || echo "Failed to download pneu-gred-7.gif"
wget -q -O $output_dir/pocus_videos/convex/Pneu-grep-pneumonia3.mp4 https://img.grepmed.com/uploads/6439/pulmonary-ultrasound-pocus-clinical-lung-original.mp4 && echo "Downloaded Pneu-grep-pneumonia3.mp4" || echo "Failed to download Pneu-grep-pneumonia3.mp4"
wget -q -O $output_dir/pocus_videos/convex/Reg-Grep-Alines.mp4 https://img.grepmed.com/uploads/7408/pocus-alines-lung-normal-ultrasound-original.mp4 && echo "Downloaded Reg-Grep-Alines.mp4" || echo "Failed to download Reg-Grep-Alines.mp4"
wget -q -O $output_dir/pocus_videos/convex/Reg-Grep-Normal.gif https://img.grepmed.com/uploads/5325/clinical-pulmonary-lung-normal-artifact-original.gif && echo "Downloaded Reg-Grep-Normal.gif" || echo "Failed to download Reg-Grep-Normal.gif"
wget -q -O $output_dir/pocus_videos/convex/Pneu-grep-shredsign-consolidation.mp4 https://img.grepmed.com/uploads/7583/lung-pocus-ultrasound-shredsign-consolidation-original.mp4 && echo "Downloaded Pneu-grep-shredsign-consolidation.mp4" || echo "Failed to download Pneu-grep-shredsign-consolidation.mp4"
wget -q -O $output_dir/pocus_videos/convex/Pneu-grep-bacterial-hepatization-clinical.mp4 https://img.grepmed.com/uploads/7582/pocus-pneumonia-bacterial-hepatization-clinical-original.mp4 && echo "Downloaded Pneu-grep-bacterial-hepatization-clinical.mp4" || echo "Failed to download Pneu-grep-bacterial-hepatization-clinical.mp4"
wget -q -O $output_dir/pocus_videos/convex/Pneu-grep-pulmonary-pneumonia.mp4 https://img.grepmed.com/uploads/6952/pocus-clinical-lung-pulmonary-pneumonia-original.mp4 && echo "Downloaded Pneu-grep-pulmonary-pneumonia.mp4" || echo "Failed to download Pneu-grep-pulmonary-pneumonia.mp4"
wget -q -O $output_dir/pocus_videos/convex/Cov-grep-7543.mp4 https://img.grepmed.com/uploads/7543/clinical-covid19-pocus-sarscov2-lung-original.mp4 && echo "Downloaded Cov-grep-7543.mp4" || echo "Failed to download Cov-grep-7543.mp4"
wget -q -O $output_dir/pocus_videos/convex/Cov-grep-7525.mp4 https://img.grepmed.com/uploads/7525/lung-ultrasound-pocus-coronavirus-covid19-original.mp4 && echo "Downloaded Cov-grep-7525.mp4" || echo "Failed to download Cov-grep-7525.mp4"
wget -q -O $output_dir/pocus_videos/convex/Cov-grep-7511.mp4 https://img.grepmed.com/uploads/7511/ultrasound-covid19-pocus-sarscov2-coronavirus-original.mp4 && echo "Downloaded Cov-grep-7511.mp4" || echo "Failed to download Cov-grep-7511.mp4"
wget -q -O $output_dir/pocus_videos/convex/Cov-grep-7510.mp4 https://img.grepmed.com/uploads/7510/clinical-blines-covid19-pocus-lung-original.mp4 && echo "Downloaded Cov-grep-7510.mp4" || echo "Failed to download Cov-grep-7510.mp4"
wget -q -O $output_dir/pocus_videos/convex/Cov-grep-7507.mp4 https://img.grepmed.com/uploads/7507/ultrasound-sarscov2-pocus-clinical-lung-original.mp4 && echo "Downloaded Cov-grep-7507.mp4" || echo "Failed to download Cov-grep-7507.mp4"
wget -q -O $output_dir/pocus_videos/convex/Cov-grep-7505.mp4 https://img.grepmed.com/uploads/7505/covid19-pocus-ultrasound-sarscov2-clinical-original.mp4 && echo "Downloaded Cov-grep-7505.mp4" || echo "Failed to download Cov-grep-7505.mp4"
wget -q -O $output_dir/pocus_videos/convex/Cov-grep-7453.mp4 https://img.grepmed.com/uploads/7453/covid19-lung-sarscov2-coronavirus-skiplesions-original.mp4 && echo "Downloaded Cov-grep-7453.mp4" || echo "Failed to download Cov-grep-7453.mp4"
wget -q -O $output_dir/pocus_videos/convex/Reg_alines_advancesVid4.mp4 https://ars.els-cdn.com/content/image/1-s2.0-S0733862715000772-mmc4.mp4 && echo "Downloaded Reg_alines_advancesVid4.mp4" || echo "Failed to download Reg_alines_advancesVid4.mp4"
wget -q -O $output_dir/pocus_videos/convex/Vir_blines_advancesVid9.mp4 https://ars.els-cdn.com/content/image/1-s2.0-S0733862715000772-mmc9.mp4 && echo "Downloaded Vir_blines_advancesVid9.mp4" || echo "Failed to download Vir_blines_advancesVid9.mp4"
wget -q -O $output_dir/pocus_videos/convex/Pneu_consol_advancesVid10.mp4 https://ars.els-cdn.com/content/image/1-s2.0-S0733862715000772-mmc10.mp4 && echo "Downloaded Pneu_consol_advancesVid10.mp4" || echo "Failed to download Pneu_consol_advancesVid10.mp4"
wget -q -O $output_dir/pocus_videos/convex/Reg_clinicalreview_mov1.mp4 https://static-content.springer.com/esm/art%3A10.1186%2Fcc5668/MediaObjects/13054_2007_5188_MOESM1_ESM.avi && echo "Downloaded Reg_clinicalreview_mov1.mp4" || echo "Failed to download Reg_clinicalreview_mov1.mp4"
wget -q -O $output_dir/pocus_videos/convex/Pneu_clinicalreview_MOV4.mp4 https://static-content.springer.com/esm/art%3A10.1186%2Fcc5668/MediaObjects/13054_2007_5188_MOESM4_ESM.avi && echo "Downloaded Pneu_clinicalreview_MOV4.mp4" || echo "Failed to download Pneu_clinicalreview_MOV4.mp4"
wget -q -O $output_dir/pocus_images/convex/Cov_blines_thoraric_paperfig1.png https://media.springernature.com/lw685/springer-static/image/art%3A10.1007%2Fs40477-020-00458-7/MediaObjects/40477_2020_458_Fig1_HTML.jpg && echo "Downloaded Cov_blines_thoraric_paperfig1.png" || echo "Failed to download Cov_blines_thoraric_paperfig1.png"
wget -q -O $output_dir/pocus_images/convex/Cov_blines_thoraric_paperfig2.png https://media.springernature.com/lw685/springer-static/image/art%3A10.1007%2Fs40477-020-00458-7/MediaObjects/40477_2020_458_Fig2_HTML.jpg && echo "Downloaded Cov_blines_thoraric_paperfig2.png" || echo "Failed to download Cov_blines_thoraric_paperfig2.png"
wget -q -O $output_dir/pocus_images/convex/Cov_pleuralthickening_thoraric_paperfig8.png https://media.springernature.com/lw685/springer-static/image/art%3A10.1007%2Fs40477-020-00458-7/MediaObjects/40477_2020_458_Fig7_HTML.jpg && echo "Downloaded Cov_pleuralthickening_thoraric_paperfig8.png" || echo "Failed to download Cov_pleuralthickening_thoraric_paperfig8.png"
wget -q -O $output_dir/pocus_images/convex/Cov_whitelungs_thoraric_paperfig5.png https://media.springernature.com/lw685/springer-static/image/art%3A10.1007%2Fs40477-020-00458-7/MediaObjects/40477_2020_458_Fig4_HTML.jpg && echo "Downloaded Cov_whitelungs_thoraric_paperfig5.png" || echo "Failed to download Cov_whitelungs_thoraric_paperfig5.png"
wget -q -O $output_dir/pocus_images/convex/Cov_pleuraleffusion_thoraric_paperfig9.png https://media.springernature.com/lw685/springer-static/image/art%3A10.1007%2Fs40477-020-00458-7/MediaObjects/40477_2020_458_Fig8_HTML.jpg && echo "Downloaded Cov_pleuraleffusion_thoraric_paperfig9.png" || echo "Failed to download Cov_pleuraleffusion_thoraric_paperfig9.png"
wget -q -O $output_dir/pocus_images/convex/Cov_subpleuralthickening_thoraric_paperfig6.png https://media.springernature.com/lw685/springer-static/image/art%3A10.1007%2Fs40477-020-00458-7/MediaObjects/40477_2020_458_Fig5_HTML.jpg && echo "Downloaded Cov_subpleuralthickening_thoraric_paperfig6.png" || echo "Failed to download Cov_subpleuralthickening_thoraric_paperfig6.png"
wget -q -O $output_dir/pocus_images/convex/Cov_efsumb3.png https://wfumb.info/wp-content/uploads/2020/04/wfumb_april_fig06-1-1024x682.png && echo "Downloaded Cov_efsumb3.png" || echo "Failed to download Cov_efsumb3.png"
wget -q -O $output_dir/pocus_images/convex/Reg_efsumb2.png https://wfumb.info/wp-content/uploads/2020/04/wfumb_april_fig01.png && echo "Downloaded Reg_efsumb2.png" || echo "Failed to download Reg_efsumb2.png"
wget -q -O $output_dir/pocus_images/convex/Pneu_sonographiebilder1.jpg https://sonographiebilder.de/fileadmin/_processed_/c/b/csm_Pneumonie_li_canifizierend_24ebb9e166.jpg && echo "Downloaded Pneu_sonographiebilder1.jpg" || echo "Failed to download Pneu_sonographiebilder1.jpg"
wget -q -O $output_dir/pocus_images/convex/Pneu_sonographiebilder2.jpg https://sonographiebilder.de/fileadmin/_migrated/pics/Pneumonie__3_.jpg && echo "Downloaded Pneu_sonographiebilder2.jpg" || echo "Failed to download Pneu_sonographiebilder2.jpg"
wget -q -O $output_dir/pocus_videos/convex/Cov_Arnthfield_2020_Vid3.mp4 https://www.medrxiv.org/content/medrxiv/early/2020/10/22/2020.10.13.20212258/DC3/embed/media-3.mp4 && echo "Downloaded Cov_Arnthfield_2020_Vid3.mp4" || echo "Failed to download Cov_Arnthfield_2020_Vid3.mp4"

echo "Files downloaded."