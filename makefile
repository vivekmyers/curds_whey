params = eps n p q rho

all: $(foreach param, $(params), ablation_$(param).png)

ablation_%.png:
	python eval.py --sweep $* $(flags)

jobs: job-0.1 job-1.0 job-2.0 job-5.0 job-10.0

job-%:
	python eval.py --sweep p --eps $* $(flags) --suffix $*


clean:
	rm -f ablation_*.png

