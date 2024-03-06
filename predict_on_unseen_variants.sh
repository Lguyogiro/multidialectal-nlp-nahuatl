for file in PAPER_ncx-story.txt  PAPER_nhw.txt
do
  python predict.py logs/combined/nhi+azz_fold\=1/2024.02.14_08.44.43/model.pt ../$file predictions_$file_nhi+azz.out --raw_text
  python predict.py logs/combined_simul_mlm_nahuatl_pretraining/simul_mlm_nhi+azz_fold\=1/2024.02.24_20.27.03/model.pt ../$file predictions_$file_nhi+azz_w-mlm-multi.out --raw_text
done
