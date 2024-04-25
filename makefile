params = eps n p q rho

all: $(foreach param, $(params), ablation_$(param)_uniform.png ablation_$(param)_gaussian.png)

ablation_%_uniform.png:
	python eval.py --param $* --beta uniform 

ablation_%_gaussian.png:
	python eval.py --param $* --beta gaussian

clean:
	rm -f ablation_*.png

