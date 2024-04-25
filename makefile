params = eps n p q rho

all: $(foreach param, $(params), ablation_$(param)_uniform.png ablation_$(param)_gaussian.png)

uniform: $(foreach param, $(params), ablation_$(param)_uniform.png)

gaussian: $(foreach param, $(params), ablation_$(param)_gaussian.png)

constant: $(foreach param, $(params), ablation_$(param)_constant.png)

ablation_%_uniform.png:
	python eval.py --param $* --beta uniform 

ablation_%_gaussian.png:
	python eval.py --param $* --beta gaussian

ablation_%_constant.png:
	python eval.py --param $* --beta constant

clean:
	rm -f ablation_*.png

