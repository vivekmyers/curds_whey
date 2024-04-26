params = eps n p q rho

all: $(foreach param, $(params), ablation_$(param).png)

ablation_%.png:
	python eval.py --sweep $* $(flags)

clean:
	rm -f ablation_*.png

