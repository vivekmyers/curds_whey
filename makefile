params = eps n p q rho

all: $(foreach param, $(params), ablation_$(param)_uniform.png ablation_$(param)_gaussian.png ablation_$(param)_constant.png)

uniform: $(foreach param, $(params), ablation_$(param)_uniform.png)

gaussian: $(foreach param, $(params), ablation_$(param)_gaussian.png)

constant: $(foreach param, $(params), ablation_$(param)_constant.png)

shift: $(foreach param, $(params), ablation_$(param)_shift.png)

ablation_%_uniform.png:
	python eval.py --param $* --beta uniform $(flags)

ablation_%_gaussian.png:
	python eval.py --param $* --beta gaussian $(flags)

ablation_%_constant.png:
	python eval.py --param $* --beta constant $(flags)

ablation_%_shift.png:
	python eval.py --param $* --beta shifted_gaussian $(flags)

%_fixed.png:
	$(MAKE) flags="--fixed" $*.png

clean:
	rm -f ablation_*.png

