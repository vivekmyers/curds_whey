params = eps n p q rho pq beta0 beta0_2 beta0_1 rho_1 rho_2

all: $(foreach param, $(params), ablation_$(param).png) ablation_pq_snr.png ablation_p_snr.png

ablation_%_snr.pkl:
	$(MAKE) run-$*-pkl flags+="--curds_only --suffix snr --mode plot"
	
ablation_%.pkl:
	$(MAKE) run-$*-pkl 

ablation_%_snr.png: ablation_%_snr.pkl
	$(MAKE) run-$*-png flags+="--curds_only --suffix snr --mode plot"

ablation_%.png: ablation_%.pkl
	$(MAKE) run-$*-png 

run-%-png:
	python eval.py --sweep $* $(flags) --mode plot

run-%-pkl:
	python eval.py --sweep $* $(flags) --mode compute


ablation_beta0_1.png: ablation_beta0_1.pkl
ablation_beta0_1.%:
	$(MAKE) run-beta0-$* flags+="--p 50 --q 20 --rho 0.3 --eps 1.0 --n 100 --suffix 1"

ablation_beta0_2.png: ablation_beta0_2.pkl
ablation_beta0_2.%:
	$(MAKE) run-beta0-$* flags+="--p 50 --q 20 --rho 0.9 --eps 1.0 --n 100 --suffix 2"

ablation_rho_1.png: ablation_rho_1.pkl
ablation_rho_1.%:
	$(MAKE) run-rho-$* flags+="--p 50 --q 20 --rho 0.3 --eps 1.0 --n 100 --beta0 0.1 --suffix 1"

ablation_rho_2.png: ablation_rho_2.pkl
ablation_rho_2.%:
	$(MAKE) run-rho-$* flags+="--p 50 --q 20 --rho 0.3 --eps 1.0 --n 100 --beta0 0.9 --suffix 2"

clean:
	rm -f ablation_*.png ablation_*.pkl

overleaf:
	cp ablation_*.png overleaf/figures/

.SECONDARY:
