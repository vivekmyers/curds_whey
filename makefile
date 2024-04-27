params = eps n p q rho pq

all: $(foreach param, $(params), ablation_$(param).png) ablation_pq_snr.png ablation_p_snr.png

ablation_%_snr.png:
	python eval.py --sweep $* --curds_only $(flags)
	
ablation_%.png:
	python eval.py --sweep $* $(flags)

clean:
	rm -f ablation_*.png

