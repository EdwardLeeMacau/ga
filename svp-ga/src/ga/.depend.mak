chromosome.o: chromosome.cpp global.h chromosome.h myrand.h mt19937ar.h
myrand.o: myrand.cpp myrand.h mt19937ar.h
global.o: global.cpp global.h chromosome.h myrand.h mt19937ar.h
mt19937ar.o: mt19937ar.cpp
svpchromosome.o: svpchromosome.cpp svpchromosome.h chromosome.h
ga.o: ga.cpp global.h chromosome.h myrand.h mt19937ar.h statistics.h ga.h \
 svpchromosome.h
