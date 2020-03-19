main: main.o yonogwdnn.o parkind1.o nogwdnn_mod.o 
	$(FC) -o $@ $^ -L/usr/lib64/libblas.so -lblas
	
main.o: parkind1.o yonogwdnn.o nogwdnn_mod.o
nogwdnn_mod.o: parkind1.o yonogwdnn.o

%.o: %.F90
	$(FC) -c $< -o $(basename $<).o
	
.PHONY: clean
clean:
	rm -f *.o *.mod main