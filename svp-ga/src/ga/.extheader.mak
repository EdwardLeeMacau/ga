ga.d: ../../include/mt19937ar.h ../../include/myrand.h ../../include/global.h ../../include/statistics.h ../../include/chromosome.h ../../include/svpchromosome.h ../../include/ga.h 
../../include/mt19937ar.h: mt19937ar.h
	@rm -f ../../include/mt19937ar.h
	@ln -fs ../src/ga/mt19937ar.h ../../include/mt19937ar.h
../../include/myrand.h: myrand.h
	@rm -f ../../include/myrand.h
	@ln -fs ../src/ga/myrand.h ../../include/myrand.h
../../include/global.h: global.h
	@rm -f ../../include/global.h
	@ln -fs ../src/ga/global.h ../../include/global.h
../../include/statistics.h: statistics.h
	@rm -f ../../include/statistics.h
	@ln -fs ../src/ga/statistics.h ../../include/statistics.h
../../include/chromosome.h: chromosome.h
	@rm -f ../../include/chromosome.h
	@ln -fs ../src/ga/chromosome.h ../../include/chromosome.h
../../include/svpchromosome.h: svpchromosome.h
	@rm -f ../../include/svpchromosome.h
	@ln -fs ../src/ga/svpchromosome.h ../../include/svpchromosome.h
../../include/ga.h: ga.h
	@rm -f ../../include/ga.h
	@ln -fs ../src/ga/ga.h ../../include/ga.h
