params = eps n p q rho pq

all: $(foreach param, $(params), ablation_$(param).png) ablation_pq_snr.png ablation_p_snr.png

ablation_%_snr.pkl:
	python eval.py --sweep $* --curds_only $(flags) --suffix snr --noplot
	
ablation_%.pkl:
	python eval.py --sweep $* $(flags) --noplot

ablation_%_snr.png: ablation_%_snr.pkl
	python eval.py --sweep $* --curds_only $(flags) --suffix snr --nocompute
	
ablation_%.png: ablation_%.pkl
	python eval.py --sweep $* $(flags) --nocompute

clean:
	rm -f ablation_*.png ablation_*.pkl

.SECONDARY:
